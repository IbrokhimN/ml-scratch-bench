// Entry point of the CUDA benchmark.
// Runs every algorithm for each dataset size and writes the results to a CSV file.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "common/csv_writer.hpp"
#include "common/cuda_utils.cuh"
#include "common/data_gen.hpp"
#include "common/rng.hpp"
#include "models/gmm_em.cuh"
#include "models/kernel_pca.cuh"
#include "models/mlp.cuh"
#include "models/random_forest.cuh"
#include "optimizers/adamw.cuh"
#include "optimizers/lbfgs.cuh"
#include "optimizers/nadam.cuh"
#include "optimizers/rmsprop.cuh"
#include "optimizers/sgd_nesterov.cuh"
#include "optimizers/sgdr.cuh"

int main(int argc, char* argv[]) {
    int dev_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if(dev_count==0){ std::cerr<<"[CUDA] No GPU found!\n"; return 1; }
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout<<"[CUDA] Device: "<<prop.name<<"\n";

    std::string out=(argc>1)?argv[1]:"results/cuda_results.csv";
    G_CSV.open(out);
    if(!G_CSV){ std::cerr<<"Cannot open "<<out<<"\n"; return 1; }
    G_CSV<<"algorithm,n_samples,n_features,time_ms,metric_name,metric_value,device\n";
    G_CSV<<std::fixed;

    // Dataset sizes to sweep and the fixed hyperparameters
    std::vector<int> sizes={512,1024,2048,4096,8192,16384};
    const int FEAT=32, EPOCHS=60, BATCH=64, K_CLS=4;

    for(int n:sizes){
        std::cout<<"[CUDA] n="<<n<<"\n";

        std::vector<float> X_cls; std::vector<int> y_cls;
        gen_multiclass(X_cls,y_cls,n,FEAT,K_CLS);
        // Binary labels are the parity of the class id
        std::vector<int> y_bin(n); for(int i=0;i<n;++i) y_bin[i]=(y_cls[i]%2);

        // Each block below runs one algorithm and writes its metrics
        std::cout<<"  AdamW ... "<<std::flush;
        { auto r=adamw_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("AdamW",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms, acc="<<r.accuracy<<"%\n"; }

        std::cout<<"  Nadam ... "<<std::flush;
        { auto r=nadam_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("Nadam",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("Nadam",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  RMSProp ... "<<std::flush;
        { auto r=rmsprop_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("RMSProp",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("RMSProp",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  SGD_Nesterov ... "<<std::flush;
        { auto r=sgd_nesterov_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  SGDR ... "<<std::flush;
        { auto r=sgdr_cuda(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGDR",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGDR",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  L-BFGS ... "<<std::flush;
        { auto r=lbfgs_cuda(X_cls,y_bin,n,FEAT,50,10);
          write_row("LBFGS",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("LBFGS",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  GMM-EM ... "<<std::flush;
        { auto r=gmm_em_cuda(X_cls,n,FEAT,K_CLS,80);
          write_row("GMM_EM",n,FEAT,r.time_ms,"LogLikelihood",r.log_likelihood);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  Kernel PCA ... "<<std::flush;
        { auto r=kpca_cuda(X_cls,n,FEAT,std::min(n,128),1.f,30);
          write_row("KernelPCA",n,FEAT,r.time_ms,"VarExplained",r.variance_explained);
          std::cout<<r.time_ms<<" ms\n"; }

        std::cout<<"  MLP-AdamW ... "<<std::flush;
        { auto r=mlp_adamw_cuda(X_cls,y_cls,n,FEAT,K_CLS,EPOCHS,BATCH);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"CE_Loss",r.final_loss);
          std::cout<<r.time_ms<<" ms\n"; }

        // The forest is capped at 4096 samples to match the CPU run
        std::cout<<"  RandomForest ... "<<std::flush;
        { int n_rf=std::min(n,4096);
          std::vector<float> Xrf(X_cls.begin(),X_cls.begin()+n_rf*FEAT);
          std::vector<int>   yrf(y_cls.begin(),y_cls.begin()+n_rf);
          auto r=rf_cuda(Xrf,yrf,n_rf,FEAT,K_CLS,20,8,4);
          write_row("RandomForest",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          std::cout<<r.time_ms<<" ms\n"; }
    }

    G_CSV.close();
    std::cout<<"\n[CUDA] Results written to "<<out<<"\n";
    return 0;
}
