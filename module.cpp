#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cmath>
// Uncomment for ISPC
//#include "module_ispc.h"
//using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int &x, int &y, const int &sizeX) {
    // Note that sizeX is the size of a Row, not the number of rows
    return tensor[x * (sizeX)+ y];
}

inline void twoDimWrite(std::vector<float> &tensor, int &x, int &y, const int &sizeX, float &val) {
    tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ) {
    return tensor[x*sizeX*sizeY*sizeZ+y*sizeY*sizeZ+z*sizeZ+b];
}

inline void fourDimWrite(std::vector<float> &tensor, int &x, int &y, int &z, int &b, 
        const int &sizeX, const int &sizeY, const int &sizeZ, float &val) {
    tensor[x*sizeX*sizeY*sizeZ+y*sizeY*sizeZ+z*sizeZ+b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
    tensor = tensor.flatten();
    tensor = tensor.contiguous();
    std::vector<float> vec(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel());
    return vec;
}

/* Programming Your Attention Modules.
 * 
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We have also created O and QK^t Tensors 
 * that are formatted as vectors. After you have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it this way - if I asked my dnn
 * a question and it output 5 different answers it had a batch size of 5. These samples are independent of each
 * other and thus can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This effectively allows each head
 * to operate the same attention algorithm, but each with each head using different hyperparameters. These
 * allow each head to have their own definition of what relevance is when looking at a token. These heads
 * can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a capital letters). The
 * emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)
    
    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);
    
    /* Here is an example of how to read/write 0's to  Q (B, H, N, d) using the 4D accessors

        //loop over Batch Size
         for (int b = 0; b < B; b++) {

             //loop over Heads
             for (int h = 0; h < H; h++) {

                 //loop over Sequence Length
                 for (int i = 0; i < N; i++) {

                     //loop over Embedding Dimensionality
                     for (int j = 0; j < d; j++) {
                        float val = fourDimRead(Q, b, h, i, j, H, N, d);
                        val = 0.0;
                        fourDimWrite(Q, b, h, i, j, H, N, d, val);
                     }
                 }
             }
         }
    */

    /* Here is an example of how to read/write 0's to  QK_t (N, N) using the 2D accessors

           for (int i = 0; i < N; i++) {
	       for (int j = 0; j < N; j++) {
	           float val = twoDimRead(QK_t, i, j, N);
               val = 0.0;
	           twoDimWrite(QK_t, i, j, N, val);
             }
         }
    */
    
    // -------- YOUR CODE HERE  -------- //
    //cal QK_t
    
    // printf("B:%d\tH:%d\tN:%d\td:%d\n",B,H,N,d);
    for(int b=0;b<B;b++){
        for(int h=0;h<H;h++){
            // printf("before QK_T\n");
            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    float tmp = 0.0f;
                    for(int k=0;k<d;k++){
                        // tmp += fourDimRead(Q,b,h,i,k,H,N,d)*fourDimRead(K,b,h,j,k,H,N,d);
                        // printf("before read\n");
                        tmp += fourDimRead(Q,b,h,i,k,H,N,d)*fourDimRead(K,b,h,j,k,H,N,d);
                    }
                    // printf("before write\n");
                    twoDimWrite(QK_t,i,j,N,tmp);
                }
            }
            // printf("before softmax\n");
            for(int i=0;i<N;i++){
                float down = 0.0f;
                for(int j=0;j<N;j++){
                    down += exp(twoDimRead(QK_t,i,j,N));
                }
                for(int j=0;j<N;j++){
                    float tmp = exp(twoDimRead(QK_t,i,j,N)) / down;
                    twoDimWrite(QK_t,i,j,N,tmp);
                }
            }
            // printf("before Q\n");
            for(int i=0;i<N;i++){
                for(int j=0;j<d;j++){
                    float tmp = 0.0f;
                    for(int k=0;k<N;k++){
                        tmp += twoDimRead(QK_t,i,k,N)*fourDimRead(V,b,h,k,j,H,N,d);
                    }
                    fourDimWrite(O,b,h,i,j,H,N,d,tmp);
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor QK_tTensor,
                int B, int H, int N, int d){
    
    // Q, K, V are passed in with Shape: (B, H, N, d)
    //QK^t Intermediate Tensor has Shape (N, N)

    //Make O Tensor with Shape (B, H, N, d) 
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

    //Format O, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);

    //Format QK_t Tensor into a 2D vector.
    std::vector<float> QK_t = formatTensor(QK_tTensor);

    int blocksize = 16;  //tile size //16: almost 2x accelorate(if don't consider align,which not use min function)
                                    //16: 1.3x accelorate(use min function)
    // -------- YOUR CODE HERE  -------- //
    printf("B:%d\tH:%d\tN:%d\td:%d\n",B,H,N,d);
    for(int b=0;b<B;b++){
        for(int h=0;h<H;h++){
            printf("before QK_T\n");
            //blocking
            for(int i=0;i<N;i+=blocksize){
                for(int j=0;j<N;j+=blocksize){
                    //for each small matrix mul
                    for(int t=0;t<d;t+=blocksize){
                        //for each element in the small matrix
                        for(int i_ = i;i_ < std::min(N,i+blocksize); i_++){
                            for(int j_ = j; j_ < std::min(N,j+blocksize); j_++){
                                float tmp = 0.f;
                                // int idx = i+i_;
                                // int idx = i+i_;
                                // int jdx = j+j_;
                                // if(idx>=N || jdx>=N) continue;
                                // int jdx = j+j_;
                                for(int k_ = t;k_ < std::min(d,t+blocksize); k_++){
                                    // int kdx = k_ + t;
                                    // if(kdx>=d) break;
                                    tmp += fourDimRead(Q,b,h,i_,k_,H,N,d) * fourDimRead(K,b,h,j_,k_,H,N,d);
                                }
                                tmp += twoDimRead(QK_t,i_,j_,N);
                                twoDimWrite(QK_t,i_,j_,N,tmp);
                            }
                        }
                    }
                }
            }

            // printf("before softmax\n");
            for(int i=0;i<N;i++){
                float down = 0.0f;
                for(int j=0;j<N;j++){
                    down += exp(twoDimRead(QK_t,i,j,N));
                }
                for(int j=0;j<N;j++){
                    float tmp = exp(twoDimRead(QK_t,i,j,N)) / down;
                    twoDimWrite(QK_t,i,j,N,tmp);
                }
            }
            // printf("before O\n");
            for(int i=0;i<N;i+=blocksize){
                for(int j=0;j<d;j+=blocksize){
                    for(int t = 0;t<N;t+=blocksize){
                        for(int i_=i;i_<std::min(N,i+blocksize);i_++){
                            for(int j_=j;j_<std::min(d,j+blocksize);j_++){
                                float tmp = 0.0f;
                                // int idx = i+i_;
                                // int jdx = j+j_;
                                // if(idx>=N || jdx>=d) continue;
                                for(int k_=t;k_<std::min(N,t+blocksize);k_++){
                                    // int kdx = k_ + t;
                                    // if(kdx>=N) break;
                                    tmp += twoDimRead(QK_t,i_,k_,N)*fourDimRead(V,b,h,k_,j_,H,N,d);
                                }
                                tmp += fourDimRead(O,b,h,i_,j_,H,N,d);
                                fourDimWrite(O,b,h,i_,j_,H,N,d,tmp);
                            }
                        }
                    }
                }
            }
        }
    }

    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

//if not use openmp,it will be a little slower than part1 with memory use 10x less than part1
//use openmp,it will be almost 5x faster than part1.(num_thread is 8)
torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, torch::Tensor temp,
                int B, int H, int N, int d){

    // Q, K, V are passed in with Shape: (B, H, N, d)

    //Make O Tensor with Shape (B, H, N, d)
    //and O Row Tensor with Shape (N)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
    at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

    //Format Y, Q, K, and V tensors into 4D vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    
    //Format ORow Tensor into a 1D vector
    // You can simply access this as ORow[i]
    std::vector<float> ORow = formatTensor(ORowTensor);


    // -------- YOUR CODE HERE  -------- //
    // We give you a template of the first three loops for your convenience
    //loop over batch
    #pragma omp parallel for collapse(3)  
    for (int b = 0; b < B; b++){

        //loop over heads
        for (int h = 0; h < H; h++){
            for (int i = 0; i < N ; i++){

		// YRow is moved inside so each OpenMP thread gets a local copy.
                at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});      
                std::vector<float> ORow = formatTensor(ORowTensor);
		//YOUR CODE HERE
                //get a row of QK_T 
                for(int j=0;j<N;j++){
                    float tmp = 0.0f;
                    for(int k=0;k<d;k++){
                        tmp += fourDimRead(Q,b,h,i,k,H,N,d)*fourDimRead(K,b,h,j,k,H,N,d);
                    }
                    ORow[j] = tmp;
                }
                //softmax one row
                float down = 0.0f;
                for(int j=0;j<N;j++){
                    down += exp(ORow[j]);
                }
                for(int j=0;j<N;j++){
                    ORow[j] = exp(ORow[j]) / down;
                }
                //get a row of output
                for(int j=0;j<d;j++){
                    float tmp = 0.0f;
                    for(int k=0;k<N;k++){
                        tmp += ORow[k] * fourDimRead(V,b,h,k,j,H,N,d);
                    }
                    fourDimWrite(O,b,h,i,j,H,N,d,tmp);
                }
            }
	}
    }
	    
	
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
               torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
               torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
               torch::Tensor OiTensor, torch::Tensor LTensor,  torch::Tensor LiTensor, 
	       torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
                int B, int H, int N, int d) {
        
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)

    //Make O Tensor with Shape (B, H, N, d)
    at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
   
    //Format All Tensors into Vectors
    std::vector<float> O = formatTensor(OTensor);
    std::vector<float> Q = formatTensor(QTensor);
    std::vector<float> K = formatTensor(KTensor);
    std::vector<float> V = formatTensor(VTensor);
    std::vector<float> Sij = formatTensor(SijTensor);
    std::vector<float> Pij = formatTensor(PijTensor);
    std::vector<float> Kj = formatTensor(KjTensor);
    std::vector<float> Vj = formatTensor(VjTensor);
    std::vector<float> Qi = formatTensor(QiTensor);
    std::vector<float> Oi = formatTensor(OiTensor);
    std::vector<float> l = formatTensor(LTensor);
    std::vector<float> PV = formatTensor(PVTensor);
    std::vector<float> li = formatTensor(LiTensor);
    std::vector<float> lij = formatTensor(LijTensor);
    std::vector<float> lnew = formatTensor(LnewTensor);

    // -------- YOUR CODE HERE  -------- //
    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;
    // printf("B:%d\tH:%d\tN:%d\td:%d\n",B,H,N,d);
    // printf("Br:%d\tBc:%d\n",Br,Bc);
    // printf("Tr:%d\tTc:%d\n",Tr,Tc);
    // printf("print input Q:\n");
    // for(int b=0;b<B;b++){
    //     for(int h=0;h<H;h++){
    //         for(int i=0;i<N;i++){
    //             for(int j=0;j<d;j++){
    //                 printf("%.4f\t",fourDimRead(Q,b,h,i,j,H,N,d));
    //             }
    //             printf("\n");
    //         }
    //         printf("------------next head------------------\n");
    //     }
    //     printf("****************next batch********************\n");
    // }
    // printf("\nprint input K:\n");
    // for(int b=0;b<B;b++){
    //     for(int h=0;h<H;h++){
    //         for(int i=0;i<N;i++){
    //             for(int j=0;j<d;j++){
    //                 printf("%.4f\t",fourDimRead(K,b,h,i,j,H,N,d));
    //             }
    //             printf("\n");
    //         }
    //         printf("------------next head------------------\n");
    //     }
    //     printf("****************next batch********************\n");
    // }
    // printf("\nprint input V:\n");
    // for(int b=0;b<B;b++){
    //     for(int h=0;h<H;h++){
    //         for(int i=0;i<N;i++){
    //             for(int j=0;j<d;j++){
    //                 printf("%.4f\t",fourDimRead(V,b,h,i,j,H,N,d));
    //             }
    //             printf("\n");
    //         }
    //         printf("------------next head------------------\n");
    //     }
    //     printf("****************next batch********************\n");
    // }
    for (int b = 0; b < B; b++){
        for (int h = 0; h < H; h++){
            std::vector<float> Sij = formatTensor(SijTensor);  //important!!!!!!!!!!!
            std::vector<float> Pij = formatTensor(PijTensor);  //this intermediate variable must be define 
            std::vector<float> Kj = formatTensor(KjTensor);    //inside the h and b loop!!!!! 
            std::vector<float> Vj = formatTensor(VjTensor);    //while O Q K V should define outside.
            std::vector<float> Qi = formatTensor(QiTensor);
            std::vector<float> Oi = formatTensor(OiTensor);
            std::vector<float> l = formatTensor(LTensor);
            std::vector<float> PV = formatTensor(PVTensor);
            std::vector<float> li = formatTensor(LiTensor);
            std::vector<float> lij = formatTensor(LijTensor);
            std::vector<float> lnew = formatTensor(LnewTensor);
            // Sij.clear();  // clear will error! 
            // Pij.clear();
            // Kj.clear();
            // Vj.clear();
            // Qi.clear();
            // Oi.clear();
            // l.clear();
            // PV.clear();
            // li.clear();
            // lij.clear();
            // lnew.clear();
            for(int j=0;j<Tc;j++){
                //load Kj,Vj
                for(int m=0;m<Bc;m++){
                    for(int n=0;n<d;n++){
                        int jBc = j*Bc + m;
                        float tmp1 = fourDimRead(K,b,h,jBc,n,H,N,d);
                        float tmp2 = fourDimRead(V,b,h,jBc,n,H,N,d);
                        twoDimWrite(Kj,m,n,d,tmp1);
                        twoDimWrite(Vj,m,n,d,tmp2);
                    }
                }

                for(int i=0;i<Tr;i++){
                    //load Qi,Oi,li
                    for(int m=0;m<Br;m++){
                        int iBr = i*Br + m;
                        for(int n=0;n<d;n++){
                            float tmp1 = fourDimRead(Q,b,h,iBr,n,H,N,d);
                            float tmp2 = fourDimRead(O,b,h,iBr,n,H,N,d);
                            twoDimWrite(Qi,m,n,d,tmp1);
                            twoDimWrite(Oi,m,n,d,tmp2);
                        }
                        li[m] = l[iBr];
                    }
                    //Sij = Qi * Kj'
                    for(int m=0;m<Br;m++){
                        for(int n=0;n<Bc;n++){
                            float tmp = 0.0f;
                            for(int k=0;k<d;k++){
                                tmp += twoDimRead(Qi,m,k,d) * twoDimRead(Kj,n,k,d);
                            }
                            twoDimWrite(Sij,m,n,Bc,tmp);
                        }
                    }
                    //Pij = exp(Sij)
                    for(int m=0;m<Br;m++){
                        for(int n=0;n<Bc;n++){
                            float tmp = exp(twoDimRead(Sij,m,n,Bc));
                            twoDimWrite(Pij,m,n,Bc,tmp);
                        }
                    }
                    //lij = rowsum(Pij)
                    for(int m=0;m<Br;m++){
                        float tmp = 0.0f;
                        for(int n=0;n<Bc;n++){
                            tmp += twoDimRead(Pij,m,n,Bc);
                        }
                        lij[m] = tmp;
                    }
                    //lnew = li + lij
                    for(int m=0;m<Br;m++){
                        lnew[m] = li[m] + lij[m];
                    }
                    //Oi = (liOi + PijVj) / lnew  (Pij * Vj, liOi is elementwise multiplication)
                    for(int m=0;m<Br;m++){
                        for(int n=0;n<d;n++){
                            float tmp = 0.0f;
                            for(int k=0;k<Bc;k++){
                                tmp += twoDimRead(Pij,m,k,Bc) * twoDimRead(Vj,k,n,d);
                            }
                            tmp += li[m] * twoDimRead(Oi,m,n,d);
                            tmp /= lnew[m];
                            twoDimWrite(Oi,m,n,d,tmp);
                        }
                    }

                    //write block Oi and lnew back to O and l in main memory
                    for(int m=0;m<Br;m++){
                        int iBr = i*Br + m;
                        for(int n=0;n<d;n++){
                            float tmp1 = twoDimRead(Oi,m,n,d);
                            fourDimWrite(O,b,h,iBr,n,H,N,d,tmp1);
                        }
                        l[iBr] = lnew[m];
                    }
                }
            }
        }
    }
    // printf("\n\n *******output*********\n");
    // for(int b=0;b<B;b++){
    //     for(int h=0;h<H;h++){
    //         for(int i=0;i<N;i++){
    //             for(int j=0;j<d;j++){
    //                 printf("%.4f\t",fourDimRead(O,b,h,i,j,H,N,d));
    //             }
    //             printf("\n");
    //         }
    //         printf("------------next head------------------\n");
    //     }
    //     printf("****************next batch********************\n");
    // }
    // DO NOT EDIT THIS RETURN STATEMENT //
    // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and returns it //
    return torch::from_blob(O.data(), {B, H, N, d}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}


/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked, " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
