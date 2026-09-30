// Entry point of the CPU benchmark.
// Runs every algorithm for each dataset size and writes the results to a CSV file.

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "common/csv_writer.hpp"
#include "common/data_gen.hpp"
#include "common/rng.hpp"
#include "common/timer.hpp"
#include "models/gmm_em.hpp"
#include "models/kernel_pca.hpp"
#include "models/mlp.hpp"
#include "models/random_forest.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/lbfgs.hpp"
#include "optimizers/nadam.hpp"
#include "optimizers/rmsprop.hpp"
#include "optimizers/sgd_nesterov.hpp"
#include "optimizers/sgdr.hpp"

int main(int argc, char* argv[]) {
    std::string out = (argc>1) ? argv[1] : "results/cpu_results.csv";
    G_CSV.open(out);
    if(!G_CSV){ std::cerr<<"Cannot open "<<out<<"\n"; return 1; }
    G_CSV << "algorithm,n_samples,n_features,time_ms,metric_name,metric_value,device\n";
    G_CSV << std::fixed;

    // Dataset sizes to sweep and the fixed hyperparameters
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192, 16384};
    const int FEAT   = 32;
    const int EPOCHS = 60;
    const int BATCH  = 64;
    const int K_CLS  = 4;

    for(int n : sizes) {
        std::cout << "[CPU] n=" << n << "\n";

        std::vector<float> X_reg; std::vector<float> y_reg;
        std::vector<float> X_cls; std::vector<int>   y_cls;
        // The regression set is not used by any algorithm.
        // It is still generated so the random stream matches earlier results.
        gen_regression(X_reg, y_reg, n, FEAT);
        gen_multiclass(X_cls, y_cls, n, FEAT, K_CLS);

        // Binary labels are the parity of the class id
        std::vector<int> y_bin(n);
        for(int i=0;i<n;++i) y_bin[i]=(y_cls[i]%2);

        // Each block below runs one algorithm and writes its metrics
        std::cout << "  AdamW ... " << std::flush;
        { auto r=adamw_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("AdamW",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms, acc=" << r.accuracy << "%\n"; }

        std::cout << "  Nadam ... " << std::flush;
        { auto r=nadam_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("Nadam",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("Nadam",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  RMSProp ... " << std::flush;
        { auto r=rmsprop_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("RMSProp",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("RMSProp",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  SGD_Nesterov ... " << std::flush;
        { auto r=sgd_nesterov_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGD_Nesterov",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  SGDR ... " << std::flush;
        { auto r=sgdr_logistic(X_cls,y_bin,n,FEAT,EPOCHS,BATCH);
          write_row("SGDR",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("SGDR",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  L-BFGS ... " << std::flush;
        { auto r=lbfgs_logistic(X_cls,y_bin,n,FEAT,50,10);
          write_row("LBFGS",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("LBFGS",n,FEAT,r.time_ms,"BCE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  GMM-EM ... " << std::flush;
        { auto r=gmm_em(X_cls,n,FEAT,K_CLS,80);
          write_row("GMM_EM",n,FEAT,r.time_ms,"LogLikelihood",r.log_likelihood);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  Kernel PCA ... " << std::flush;
        { auto r=kernel_pca(X_cls,n,FEAT,std::min(n,128),1.0f,30);
          write_row("KernelPCA",n,FEAT,r.time_ms,"VarExplained",r.variance_explained);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << "  MLP-AdamW ... " << std::flush;
        { auto r=mlp_adamw(X_cls,y_cls,n,FEAT,K_CLS,EPOCHS,BATCH);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("MLP_AdamW",n,FEAT,r.time_ms,"CE_Loss",r.final_loss);
          std::cout << r.time_ms << " ms\n"; }

        // The forest is capped at 4096 samples because tree building is slow
        std::cout << "  RandomForest ... " << std::flush;
        { int n_trees=20, mxd=8;
          int n_rf = std::min(n, 4096);
          std::vector<float> X_rf(X_cls.begin(), X_cls.begin()+n_rf*FEAT);
          std::vector<int>   y_rf(y_cls.begin(), y_cls.begin()+n_rf);
          auto r=random_forest(X_rf,y_rf,n_rf,FEAT,K_CLS,n_trees,mxd,4);
          write_row("RandomForest",n,FEAT,r.time_ms,"Accuracy",r.accuracy);
          write_row("RandomForest",n,FEAT,r.time_ms,"OOB_Accuracy",r.oob_accuracy);
          std::cout << r.time_ms << " ms\n"; }

        std::cout << std::flush;
    }

    G_CSV.close();
    std::cout << "\n[CPU] Results written to " << out << "\n";
    return 0;
}
