#include <stdio.h>
//#define Nps 512
#define Nx 150
#define Ny 150

__device__ double index(int toy, int Nps, double *data,double *y_bin, double *y_arys, int Nevts_dat) 
{
  //Construct the matrix
  int i, j;
  
  double y;
  double ymax, ymin;
 
  
  double M[Nx*Ny];
  int Nguys = 0;
  double w =.5*( y_bin[1]-y_bin[0]);
  for (i = 0; i<Nx; i++) 
  {
    //x = x_bin[i];
   for (k = 0; k<300; k++) //teño que ter en conta os valores con diferente PK
   {
    y = y_arys[i*Nps + k + toy];
    for (j = 0; j < Ny; j++) 
    {
      ymax = y_bin[j] + w ;
      
      ymin = y_bin[j] - w;
      
      if (ymax < y || ymin > y) M[Ny*i + j] = 0.;
      else {  
        M[Ny*i + j] = 1.;
        Nguys++; 
        
      } 
    }
   }
  }
  double sf = 1.; //BALA: penso que na obtención de sf para os casos nos que o plano  non contén totalmente aos Toys indúcese un sistemático
  if (Nguys > 0.5) sf = Nevts_dat*1./Nguys;
  for (i = 0; i< Nx*Ny; i++)
  {
     M[i] = M[i]*sf + data[i] + 1;
  }
  
  double theil = 0.; /// calcular a partir de M
  double events = 0.; 
  for (i = 0; i< Nx*Ny; i++) events += M[i];      
  
  double mean=events/(Nx*Ny);
 
  double T = 0.;
  for (i = 0; i< Nx*Ny; i++) T += M[i]*logf(M[i]/mean);      
  
  T = T/mean;
  T = T/(Nx*Ny*logf(Nx*Ny));
  
  return T;  
}

__global__ void Theil(int Nps, double *out, double *data, double *y_bin, double *y_arys, int Nevts_dat) 
{
  int toy = threadIdx.x + blockDim.x * blockIdx.x; //number of toy contained in y_arys
  //  if (toy >= Nps) return; 
  out[toy] = index(toy, Nps, data, y_bin, y_arys, Nevts_dat);
  
}
