#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using Texture = id<MTLTexture>;
using Encoder = id<MTLComputeCommandEncoder>;
static void require(bool ok, NSString *message) {
    if (!ok) { fprintf(stderr,"%s\n",message.UTF8String); exit(1); }
}
enum Mode { separate, outputPair, outputTriple, spatialPairs, combined, mathTail };
static const char *names[] = {"separate","head_shuffle","tail_three","spatial_pairs","combined","algebra_tail"};

struct Bench {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString*,id<MTLComputePipelineState>> *pipelines;
    Texture rgb, features[7], residual, output;
    int width, height;

    explicit Bench(NSString *file) {
        device = MTLCreateSystemDefaultDevice();
        require(device != nil, @"Metal GPU unavailable");
        queue = [device newCommandQueue];
        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:file encoding:NSUTF8StringEncoding error:&err];
        require(src != nil, err.localizedDescription);
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:options error:&err];
        require(lib != nil, err.localizedDescription);
        pipelines = [NSMutableDictionary new];
        for (NSString *name in lib.functionNames) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:fn error:&err];
            require(state != nil, err.localizedDescription);
            require(state.maxTotalThreadsPerThreadgroup >= 128, @"16x8 group unsupported");
            pipelines[name] = state;
        }
        fprintf(stderr,"GPU %s | OS %s | 16x8 groups | RGBA16Float | float accumulators\n",
                device.name.UTF8String, NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
    }

    Texture texture(int w, int h, bool shared) {
        auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                       width:w height:h mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = shared ? MTLStorageModeShared : MTLStorageModePrivate;
        Texture tex = [device newTextureWithDescriptor:d];
        require(tex != nil, @"Texture allocation failed");
        return tex;
    }

    void setup(int w, int h) {
        width=w; height=h;
        rgb=texture(w,h,true);
        output=texture(2*w,2*h,true);
        residual=texture(w,h,false);
        for (auto &f:features) f=texture(w,h,false);
        std::mt19937 rng(97);
        std::vector<_Float16> data(size_t(w)*h*4);
        for (size_t i=0;i<data.size();++i)
            data[i]=(i%4==3) ? 1.0f : float(rng() >> 8) / 16777216.0f;
        [rgb replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:data.data() bytesPerRow:w*8];
    }

    void dispatch(Encoder e, NSString *name, NSArray<id<MTLTexture>> *textures) {
        [e setComputePipelineState:pipelines[name]];
        for (NSUInteger i=0;i<textures.count;i++) [e setTexture:textures[i] atIndex:i];
        [e dispatchThreadgroups:MTLSizeMake((width+15)/16,(height+7)/8,1)
           threadsPerThreadgroup:MTLSizeMake(16,8,1)];
    }

    void frame(id<MTLCommandBuffer> cb, Mode mode) {
        Encoder e=[cb computeCommandEncoder];  // Serial dispatch, tracked texture hazards.
        bool tiled = mode==spatialPairs || mode==combined;
        bool triple = mode==outputTriple || mode==combined || mode==mathTail;
        for (int i=0; i<(triple?6:7); ) {
            Texture src = i==0 ? rgb : features[i-1];
            if (tiled && i<6) {
                dispatch(e,[NSString stringWithFormat:@"pair%d",i],@[src,features[i],features[i+1]]);
                i+=2;
            } else {
                NSString *pattern = mode==mathTail && i>0 ? @"mathSingle%d" : @"single%d";
                dispatch(e,[NSString stringWithFormat:pattern,i],@[src,features[i]]);
                ++i;
            }
        }
        NSMutableArray *inputs=[NSMutableArray new];
        for (int i=0;i<(triple?6:7);i++) [inputs addObject:features[i]];
        if (triple || mode==outputPair) {
            [inputs addObject:rgb];
            [inputs addObject:output];
            NSString *last = triple ? @"tail" : @"headShuffle";
            dispatch(e,mode==mathTail ? @"mathTail" : last,inputs);
        } else {
            [inputs addObject:residual];
            dispatch(e,@"head",inputs);
            dispatch(e,@"shuffle",@[rgb,residual,output]);
        }
        [e endEncoding];
    }

    double sample(Mode mode, int batch) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb=[queue commandBuffer];
            for (int i=0;i<batch;i++) frame(cb,mode);
            [cb commit];
            [cb waitUntilCompleted];
            require(cb.status==MTLCommandBufferStatusCompleted, cb.error.localizedDescription);
            double ms=(cb.GPUEndTime-cb.GPUStartTime)*1000/batch;
            require(ms>0, @"GPU timestamp unavailable");
            return ms;
        }
    }

    void verify() {
        std::vector<_Float16> ref(size_t(width)*height*16), actual(ref.size());
        for (int m=0;m<6;m++) {
            sample(Mode(m),1);
            [output getBytes:actual.data() bytesPerRow:width*16 fromRegion:MTLRegionMake2D(0,0,width*2,height*2) mipmapLevel:0];
            if (m==0) ref=actual;
            size_t different=0;
            float worst=0;
            double absSum=0;
            for (size_t i=0;i<ref.size();++i) {
                float d=std::abs(float(ref[i])-float(actual[i]));
                require(std::isfinite(d), @"Non-finite output");
                absSum += d;
                different += d!=0;
                worst=std::max(worst,d);
            }
            fprintf(stderr,"check %dx%d %-14s differences=%zu max_abs=%g mean_abs=%.9g\n",width,height,names[m],different,worst,absSum/ref.size());
            require(worst<=(m==mathTail ? 0.01f : 0.0f), @"Output mismatch");
        }
    }
};

int main(int argc,char **argv) {
    @autoreleasepool {
        if (argc!=2) { fprintf(stderr,"usage: bench generated.metal\n"); return 1; }
        Bench b([NSString stringWithUTF8String:argv[1]]);
        puts("input_width,input_height,round,order,mode,gpu_ms_per_frame");
        for (auto [w,h] : {std::pair{37,23},std::pair{1280,720},std::pair{1920,1080}}) {
            b.setup(w,h);
            b.verify();
            if (w==37) continue;
            std::array<Mode,6> order={separate,outputPair,outputTriple,spatialPairs,combined,mathTail};
            std::mt19937 rng(413);
            for (int round=-5;round<30;++round) {
                std::shuffle(order.begin(),order.end(),rng);
                for (int slot=0;slot<6;++slot) {
                    Mode m=order[slot];
                    double ms=b.sample(m,3);
                    if (round>=0) printf("%d,%d,%d,%d,%s,%.6f\n",w,h,round,slot,names[m],ms);
                }
                fflush(stdout);
            }
        }
    }
}
