/*========================================================================*
 *  FSLS (Fast Solvers for Linear System)  (c) 2010-2012                  *
 *  School of Mathematics and Computational Science                       *
 *  Xiangtan University                                                   *
 *  Email: peghoty@163.com                                                *
 *========================================================================*/

/*!
 * transfer.c -- data format transfering
 *
 * Created by peghoty 2010/08/27
 * Xiangtan University
 * peghoty@163.com
 *  
 */

#include "basic.h"
#include "util.h"
#include "transfer.h"

/*!
 * \fn int fsls_CSR2FullMatrix
 * \brief Transfer a CSR matrix to a full matrix(stored in one-dimensional array row by row)
 * \param fsls_CSRMatrix *A pointer to the CSR matrix
 * \param double **full_ptr pointer to the Full matrix
 * \author peghoty
 * \date 2010/05/05  
 */
int 
fsls_CSR2FullMatrix( fsls_CSRMatrix *A, double **full_ptr )
{   
   /* CSR information of A */
   int     n   = fsls_CSRMatrixNumRows(A);
   int     m   = fsls_CSRMatrixNumCols(A);
   int    *ia  = fsls_CSRMatrixI(A);
   int    *ja  = fsls_CSRMatrixJ(A);     
   double *a   = fsls_CSRMatrixData(A);
   
   double *full = NULL;
   int size = n*m;
   int row,start;
   int k;
      
   full = fsls_CTAlloc(double, size);
  
   for (row = 0; row < n; row ++)
   {
      start = m*row;
      for (k = ia[row]; k < ia[row+1]; k ++)
      {
         full[start+ja[k]] = a[k];
      }
   }
   
   *full_ptr = full; 
    
   return (0);
}


/*!
 * \fn int fsls_Full2CSRMatrix
 * \brief Transfer a full matrix(stored in one-dimensional array row by row) to a CSR matrix 
 * \param double *full pointer to the Full matrix 
 * \param fsls_CSRMatrix **A_ptr pointer to pointer to the result CSR matrix
 * \note work for nXn full matrix
 * \date 2010/06/21
 * \author peghoty 
 */
int 
fsls_Full2CSRMatrix( int n, double *full, fsls_CSRMatrix **A_ptr )
{   
   fsls_CSRMatrix *A = NULL;
   int     *ia = NULL;
   int     *ja = NULL;
   double  *a  = NULL; 
   
   int i,j,k;
   int nz = 0;  

   ia = fsls_CTAlloc(int, n+1);
    
   ia[0] = 0; 
   for (i = 0; i < n; i ++)
   {
      for (j = 0; j < n; j ++)
      {
         k = i*n + j;
         if (full[k] != 0.0)
         {
            nz ++;
         }
      }
      ia[i+1] = nz;
   }

   ja = fsls_CTAlloc(int, nz);
   a  = fsls_CTAlloc(double, nz);
   
   nz = 0;
   for (i = 0; i < n; i ++)
   {
      for (j = 0; j < n; j ++)
      {
         k = i*n + j;
         if (full[k] != 0.0)
         {
            ja[nz] = j;
            a[nz]  = full[k];
            nz ++;
         }
      }
   } 

   A = fsls_CSRMatrixCreate(n, n, nz);
   fsls_CSRMatrixI(A) = ia;
   fsls_CSRMatrixJ(A) = ja;
   fsls_CSRMatrixData(A) = a;
   
   *A_ptr = A;
   
   return (0);
}

/*!
 * \fn int fsls_FullRect2CSRMatrix
 * \brief Transfer a full matrix(stored in one-dimensional array row by row) to a CSR matrix 
 * \param double *full pointer to the Full matrix 
 * \param fsls_CSRMatrix **A_ptr pointer to pointer to the result CSR matrix
 * \note work for nXm full matrix
 * \date 2011/11/18
 * \author peghoty 
 */
int 
fsls_FullRect2CSRMatrix( int n, int m, double *full, fsls_CSRMatrix **A_ptr )
{   
   fsls_CSRMatrix *A = NULL;
   int     *ia = NULL;
   int     *ja = NULL;
   double  *a  = NULL; 
   
   int i,j,k;
   int nz = 0;  

   ia = fsls_CTAlloc(int, n+1);
    
   ia[0] = 0; 
   for (i = 0; i < n; i ++)
   {
      for (j = 0; j < m; j ++)
      {
         k = i*m + j;
         if (full[k] != 0.0)
         {
            nz ++;
         }
      }
      ia[i+1] = nz;
   }

   ja = fsls_CTAlloc(int, nz);
   a  = fsls_CTAlloc(double, nz);
   
   nz = 0;
   for (i = 0; i < n; i ++)
   {
      for (j = 0; j < m; j ++)
      {
         k = i*m + j;
         if (full[k] != 0.0)
         {
            ja[nz] = j;
            a[nz]  = full[k];
            nz ++;
         }
      }
   } 

   A = fsls_CSRMatrixCreate(n, m, nz);
   fsls_CSRMatrixI(A) = ia;
   fsls_CSRMatrixJ(A) = ja;
   fsls_CSRMatrixData(A) = a;
   
   *A_ptr = A;
   
   return (0);
}

/*!
 * \fn void fsls_TriBand2FullMatrix
 * \brief Transfer a TriBandMatrix to a FullMatrix
 * \note A should be tridiagonal!!
 * \author peghoty
 * \date 2010/07/07 
 */ 
void
fsls_TriBand2FullMatrix( fsls_BandMatrix *A, double **full_ptr )
{
   int      n       = fsls_BandMatrixN(A);
   double  *diag    = fsls_BandMatrixDiag(A);
   double **offdiag = fsls_BandMatrixOffdiag(A);    
   double  *full    = NULL;
   
   int i,start = 0;
 
   full = fsls_CTAlloc(double, n*n); 

  /*----------------------------------------------------
   * different treatment should be applied for
   * 'n = 1' and 'n > 1'.
   *--------------------------------------------------*/
   if (n > 1)
   {   
      start = 0;
      full[start++] = diag[0];
      full[start] = offdiag[1][0];
   
      for (i = 1; i < n-1; i ++)
      {
         start = i*n + (i-1);
         full[start++] = offdiag[0][i];
         full[start++] = diag[i];
         full[start]   = offdiag[1][i];
      }
   
      start = (n-1)*n + (n-2);
      full[start++] = offdiag[0][n-1];
      full[start] = diag[n-1];
   }
   else
   {
      full[0] = diag[0];
   }   
   
   /* return */
   *full_ptr = full;
}

/*!
 * \fn int fsls_Band2FullMatrix
 * \brief Transfer a 'fsls_BandMatrix' type matrix into a full matrix.
 * \param fsls_BandMatrix *A pointer to the 'fsls_BandMatrix' type matrix
 * \param double **full_ptr pointer to full matrix
 * \date 2010/08/01
 * \author peghoty
 */
int 
fsls_Band2FullMatrix( fsls_BandMatrix *A, double **full_ptr )
{
   // some members of A
   int      n       = fsls_BandMatrixN(A);
   int      nband   = fsls_BandMatrixNband(A);
   int     *offsets = fsls_BandMatrixOffsets(A);
   double  *diag    = fsls_BandMatrixDiag(A);
   double **offdiag = fsls_BandMatrixOffdiag(A);

   double *full = NULL;
   
   int i,col;
   int band,offset;
   int begin,end;
 
  /*---------------------------------------------
   * allocate memory for full matrix 
   *--------------------------------------------*/   
   full = fsls_CTAlloc(double, n*n);
   
  /*---------------------------------------------
   * fill the diagonal entries 
   *--------------------------------------------*/
   for (i = 0; i < n; i ++)
   {
      full[i*n+i] = diag[i];
   }
   
  /*---------------------------------------------
   * fill the offdiagonal entries 
   *--------------------------------------------*/
   for (band = 0; band < nband; band ++)
   {
      offset = offsets[band];
      if (offset < 0) /* deal with the bands in the bottom left */
      {
         begin = abs(offset);
         for (i = begin,col = 0; i < n; i++, col++)
         {
            full[i*n+col] = offdiag[band][i];
         }
      }
      else /* deal with the bands in the top right */
      {
         end = n - offset;
         for (i = 0,col = offset; i < end; i++, col++)
         {
            full[i*n+col] = offdiag[band][i];
         }
      }
   } 
   
   *full_ptr = full;
   
   return 0;
}

/*!
 * \fn fsls_BandMatrix *fsls_CSR2BandMatrix
 * \brief Transfer a 'fsls_BandMatrix' type matrix into a CSRMatrix.
 * \param int nx number of nodes in X-direction.
 * \param int ny number of nodes in Y-direction.
 * \param fsls_CSRMatrix *B pointer to the pointer to CSRMatrix
 * \return a pointer to the 'fsls_BandMatrix' type matrix
 * \note Here, we require A is indeed a band matrix with less then 9 band
 *       which is based on 2D struct grid.
 * \author peghoty
 * \date 2011/05/20 
 */
fsls_BandMatrix * 
fsls_CSR2BandMatrix( int nx, int ny, fsls_CSRMatrix *B )
{
   fsls_BandMatrix *A  = NULL;

   int      ngrid   = nx*ny;
   int      nband   = 8;
    
   int     *offset  = NULL;
   double  *diag    = NULL;
   double **offdiag = NULL;

   int     *ia      = fsls_CSRMatrixI(B);
   int     *ja      = fsls_CSRMatrixJ(B);
   double  *a       = fsls_CSRMatrixData(B);
   int      i,j,k,dis;
   int      nxplus1,nyplus1,nxminus1;

   /* Check the compatibility of A and B */   
   if (ngrid != fsls_CSRMatrixNumRows(B))
   {
      printf("\n >>> Warning: in 'fsls_CSR2BandMatrix', nx*ny != ngrid! \n\n");
      exit(0);
   }
 
  /*----------------------------------------------------
   *  Generate the matrix 
   *---------------------------------------------------*/    
    
   nxplus1  = nx + 1;
   nyplus1  = ny + 1;
   nxminus1 = nx - 1;

   A = fsls_BandMatrixCreate(ngrid, nband);
   fsls_BandMatrixInitialize(A);
   fsls_BandMatrixNx(A) = nx;
   fsls_BandMatrixNy(A) = ny;
    
   offset  = fsls_BandMatrixOffsets(A);
   diag    = fsls_BandMatrixDiag(A);
   offdiag = fsls_BandMatrixOffdiag(A);

   offset[0] = -1;
   offset[1] =  1;
   offset[2] = -nx;
   offset[3] =  nx;
   offset[4] = -nxplus1;
   offset[5] = -nxminus1;
   offset[6] =  nxminus1;
   offset[7] =  nxplus1;
   
   for (i = 0; i < ngrid; i ++)
   {
      for (j = ia[i]; j < ia[i+1]; j ++)
      {
          k   = ja[j];
          dis = k - i;
          if (dis == 0)
          {
             diag[i] = a[j];
          }
          else if (dis == offset[0])
          {
             offdiag[0][i] = a[j];
          }
          else if (dis == offset[1])
          {
             offdiag[1][i] = a[j];
          }
          else if (dis == offset[2])
          {
             offdiag[2][i] = a[j];
          }
          else if (dis == offset[3])
          {
             offdiag[3][i] = a[j];
          }
          else if (dis == offset[4])
          {
             offdiag[4][i] = a[j];
          }
          else if (dis == offset[5])
          {
             offdiag[5][i] = a[j];
          }
          else if (dis == offset[6])
          {
             offdiag[6][i] = a[j];
          }
          else if (dis == offset[7])
          {
             offdiag[7][i] = a[j];
          }
      }
   } 
    
   return (A);
}

/*!
 * \fn int fsls_Band2CSRMatrix
 * \brief Transfer a 'fsls_BandMatrix' type matrix into a CSRMatrix.
 * \param fsls_BandMatrix *B pointer to the 'fsls_BandMatrix' type matrix
 * \param fsls_CSRMatrix **A_ptr pointer to the pointer to CSRMatrix
 * \author peghoty
 * \date 2010/08/04 
 */
int 
fsls_Band2CSRMatrix( fsls_BandMatrix *B, fsls_CSRMatrix **A_ptr )
{
   // some members of A
   int      n       = fsls_BandMatrixN(B);
   int      nband   = fsls_BandMatrixNband(B);
   int     *offsets = fsls_BandMatrixOffsets(B);
   double  *diag    = fsls_BandMatrixDiag(B);
   double **offdiag = fsls_BandMatrixOffdiag(B);

   fsls_CSRMatrix *A = NULL;
   int *ia = NULL;
   int *ja = NULL;
   double *a = NULL;
   
   int i;
   int col;
   int band,offset;
   int begin,end;   
   int nplus1 = n + 1;
   int nzpr = nband + 1; // nonzeros per row
   
  /*---------------------------------------------
   * Create a CSR matrix 
   *--------------------------------------------*/
   A = fsls_CSRMatrixCreate(n,n,nzpr*n);
   fsls_CSRMatrixInitialize(A);
   ia = fsls_CSRMatrixI(A);
   ja = fsls_CSRMatrixJ(A);
   a  = fsls_CSRMatrixData(A);
   
  /*---------------------------------------------
   * Generate the 'ia' array 
   *--------------------------------------------*/
   for (i = 0; i < nplus1; i ++)
   {
      ia[i] = i*nzpr;
   }

  /*---------------------------------------------
   * fill the diagonal entries 
   *--------------------------------------------*/
   for (i = 0; i < n; i ++)
   {
      a[ia[i]]  = diag[i];
      ja[ia[i]] = i;
      ia[i] ++;
   }
   
  /*---------------------------------------------
   * fill the offdiagonal entries 
   *--------------------------------------------*/
   for (band = 0; band < nband; band ++)
   {
      offset = offsets[band];
      if (offset < 0) /* deal with the bands in the bottom left */
      {
         begin = abs(offset);
         for (i = begin,col = 0; i < n; i++, col++)
         {
            a[ia[i]]  = offdiag[band][i];
            ja[ia[i]] = col;
            ia[i] ++;
         }
      }
      else /* deal with the bands in the top right */
      {
         end = n - offset;
         for (i = 0,col = offset; i < end; i++, col++)
         {
            a[ia[i]]  = offdiag[band][i];
            ja[ia[i]] = col;
            ia[i] ++;            
         }
      }
   }
   
  /*---------------------------------------------
   * regenerate the 'ia' array 
   *--------------------------------------------*/
   for (i = 0; i < nplus1; i ++)
   {
      ia[i] = i*nzpr;
   }
   
  /*---------------------------------------------
   * delete zero entries in A 
   *--------------------------------------------*/   
   A = fsls_CSRMatrixDeleteZeros(A, 0.0);
   
   *A_ptr = A;
   
   return 0;
}

/*!
 * \fn fsls_LCSRMatrix *fsls_CSR2LCSRMatrix( fsls_CSRMatrix *A )
 * \brief Transfer a 'fsls_CSRMatrix' type matrix into a 'fsls_LCSRMatrix' type matrix.
 * \param *A pointer to the 'fsls_CSRMatrix' type matrix
 * \author peghoty
 * \date 2011/01/11 
 */
fsls_LCSRMatrix *
fsls_CSR2LCSRMatrix( fsls_CSRMatrix *A )
{
   double *DATA         = fsls_CSRMatrixData(A);
   int    *IA           = fsls_CSRMatrixI(A);
   int    *JA           = fsls_CSRMatrixJ(A);
   int     num_rows     = fsls_CSRMatrixNumRows(A);
   int     num_cols     = fsls_CSRMatrixNumCols(A);
   int     num_nonzeros = fsls_CSRMatrixNumNonzeros(A);

   fsls_LCSRMatrix *B = NULL;
   int     dif;
   int    *nzdifnum = NULL;
   int    *rowstart = NULL;
   int    *rowindex = fsls_CTAlloc(int, num_rows);
   int    *ja       = fsls_CTAlloc(int, num_nonzeros);
   double *data     = fsls_CTAlloc(double, num_nonzeros);
   
   /* auxiliary arrays */
   int *nzrow    = fsls_CTAlloc(int, num_rows);
   int *counter  = NULL;
   int *invnzdif = NULL;
   
   int i,j,k;
   int maxnzrow;
   int cnt;
   
   //-----------------------------------------
   //  Generate 'nzrow' and 'maxnzrow'
   //-----------------------------------------
   
   maxnzrow = 0;
   for (i = 0; i < num_rows; i ++)
   {
      nzrow[i] = IA[i+1] - IA[i];
      if (nzrow[i] > maxnzrow) 
      {
         maxnzrow = nzrow[i];
      }
   }
   /* generate 'counter' */
   counter = fsls_CTAlloc(int, maxnzrow + 1);
   for (i = 0; i < num_rows; i ++)
   {
      counter[nzrow[i]] ++;
   }

   //--------------------------------------------
   //  Determine 'dif'
   //-------------------------------------------- 
 
   for (dif = 0, i = 0; i < maxnzrow + 1; i ++)
   {
      if (counter[i] > 0) dif ++;
   }

   //--------------------------------------------
   //  Generate the 'nzdifnum' and 'rowstart'
   //-------------------------------------------- 
   
   nzdifnum = fsls_CTAlloc(int, dif);
   invnzdif = fsls_CTAlloc(int, maxnzrow + 1);
   rowstart = fsls_CTAlloc(int, dif+1);
   rowstart[0] = 0;
   for (cnt = 0, i = 0; i < maxnzrow + 1; i ++)
   {
      if (counter[i] > 0) 
      {
         nzdifnum[cnt] = i;
         invnzdif[i] = cnt;
         rowstart[cnt+1] = rowstart[cnt] + counter[i];
         cnt ++;
      }
   }

   //--------------------------------------------
   //  Generate the 'rowindex'
   //-------------------------------------------- 

   for (i = 0; i < num_rows; i ++)
   {
      j = invnzdif[nzrow[i]];
      rowindex[rowstart[j]] = i;
      rowstart[j] ++;
   }
   /* recover 'rowstart' */
   for (i = dif; i > 0; i --)
   {
      rowstart[i] = rowstart[i-1];
   }
   rowstart[0] = 0;

   //--------------------------------------------
   //  Generate the 'data' and 'ja'
   //-------------------------------------------- 
 
   for (cnt = 0, i = 0; i < num_rows; i ++)
   {
      k = rowindex[i];
      for (j = IA[k]; j < IA[k+1]; j ++)
      {
         data[cnt] = DATA[j];
         ja[cnt] = JA[j];
         cnt ++;
      }
   }
      
   //------------------------------------------------------------
   //  Create and fill a fsls_LCSRMatrix B
   //------------------------------------------------------------ 
      
   B = fsls_LCSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   fsls_LCSRMatrixDif(B)      = dif;
   fsls_LCSRMatrixNzDifNum(B) = nzdifnum;
   fsls_LCSRMatrixRowIndex(B) = rowindex;
   fsls_LCSRMatrixRowStart(B) = rowstart;
   fsls_LCSRMatrixJ(B)        = ja;
   fsls_LCSRMatrixData(B)     = data;   

   //----------------------------
   //  Free the auxiliary arrays
   //----------------------------  
    
   fsls_TFree(nzrow);
   fsls_TFree(counter);
   fsls_TFree(invnzdif);

   return B;
}

/*!
 * \fn fsls_CSRMatrix *fsls_BSRMatrix2CSRMatrix
 * \brief Transfer a 'fsls_BSRMatrix' type matrix into a CSRMatrix.
 * \param fsls_BSRMatrix *B pointer to the 'fsls_BSRMatrix' type matrix
 * \author peghoty
 * \date 2010/10/23 
 */
fsls_CSRMatrix *
fsls_BSRMatrix2CSRMatrix( fsls_BSRMatrix *B )
{
   /* members of B */
   int     ROW = fsls_BSRMatrixROW(B);
   int     COL = fsls_BSRMatrixCOL(B);
   int     NNZ = fsls_BSRMatrixNNZ(B);    
   int     nb  = fsls_BSRMatrixNb(B);
   int    *IA  = fsls_BSRMatrixIA(B);
   int    *JA  = fsls_BSRMatrixJA(B);
   int     storage_manner = fsls_BSRMatrixStorageManner(B);
   double *val = fsls_BSRMatrixVal(B);
   
   int jump = nb*nb;
   int rowA = ROW*nb;
   int colA = COL*nb;
   int nzA  = NNZ*jump;
   
   fsls_CSRMatrix *A = NULL;
   int *ia = NULL;
   int *ja = NULL;
   double *a = NULL;
   
   int i,j,k;
   int mr,mc;
   int rowstart0,rowstart,colstart0,colstart;
   int colblock,nzperrow; 
   double *vp = NULL;
   double *ap = NULL;
   int    *jap = NULL;
  
   /* Create a CSR Matrix */
   A = fsls_CSRMatrixCreate(rowA,colA, nzA);
   fsls_CSRMatrixInitialize(A);
   ia = fsls_CSRMatrixI(A);
   ja = fsls_CSRMatrixJ(A);
   a  = fsls_CSRMatrixData(A);
   
  /*--------------------------------------------------------------------------
   * Compute the number of nonzeros per row, and after this loop,
   * ia[i],i=1:rowA, will be the number of nonzeros of the (i-1)-th row.
   *------------------------------------------------------------------------*/
   for (i = 0; i < ROW; i ++)
   {
      rowstart = i*nb + 1;
      colblock = IA[i+1] - IA[i];
      nzperrow = colblock*nb;
      for (j = 0; j < nb; j ++)
      {
         ia[rowstart+j] = nzperrow;
      }
   }
   
  /*-----------------------------------------------------
   * Generate the real 'ia' for CSR of A
   *----------------------------------------------------*/
   ia[0] = 0;
   for (i = 1; i <= rowA; i ++)
   {
      ia[i] += ia[i-1];
   }
   
  /*-----------------------------------------------------
   * Generate 'ja' and 'a' for CSR of A
   *----------------------------------------------------*/
   switch (storage_manner)
   {
      case 0: // each non-zero block elements are stored in row-major order
      {
         for (i = 0; i < ROW; i ++)
         {
            for (k = IA[i]; k < IA[i+1]; k ++)
            {
               j = JA[k];
               rowstart = i*nb;
               colstart = j*nb;
               vp = &val[k*jump];
               for (mr = 0; mr < nb; mr ++)
               {
                  ap  = &a[ia[rowstart]];
                  jap = &ja[ia[rowstart]];
                  for (mc = 0; mc < nb; mc ++)
                  {
                     *ap = *vp;
                     *jap = colstart + mc;
                     vp ++;
                     ap ++;
                     jap ++;
                  }
                  ia[rowstart] += nb;
                  rowstart ++;
               }
            }
         }
      }
      break;
      
      case 1: // each non-zero block elements are stored in column-major order
      {
         for (i = 0; i < ROW; i ++)
         {
            for (k = IA[i]; k < IA[i+1]; k ++)
            {
               j = JA[k];
               rowstart0 = i*nb;
               colstart0 = j*nb;
               vp = &val[k*jump];
               for (mc = 0; mc < nb; mc ++)
               {
                  rowstart = rowstart0;
                  colstart = colstart0 + mc;
                  for (mr = 0; mr < nb; mr ++)
                  {
                     a[ia[rowstart]] = *vp; 
                     ja[ia[rowstart]] = colstart; 
                     vp ++;
                     ia[rowstart]++;
                     rowstart++;
                  }
               }
            }
         }
      }
      break;
   }
   
  /*-----------------------------------------------------
   * Map back the real 'ia' for CSR of A
   *----------------------------------------------------*/
   for (i = rowA; i > 0; i --)
   {
      ia[i] = ia[i-1];
   }
   ia[0] = 0; 
 
   return (A);   
}


/*!
 * \fn fsls_BSRMatrix *fsls_STRMatrix2BSRMatrix
 * \brief Transfer a 'fsls_STRMatrix' type matrix into a BSRMatrix.
 * \param fsls_STRMatrix *B pointer to the 'fsls_STRMatrix' type matrix
 * \author peghoty
 * \date 2010/10/26 
 */
fsls_BSRMatrix *
fsls_STRMatrix2BSRMatrix( fsls_STRMatrix *B )
{
   // members of 'B'
   int      nc      = fsls_STRMatrixNc(B);
   int      ngrid   = fsls_STRMatrixNgrid(B);
   double  *diag    = fsls_STRMatrixDiag(B);
   int      nband   = fsls_STRMatrixNband(B);
   int     *offsets = fsls_STRMatrixOffSets(B);
   double **offdiag = fsls_STRMatrixOffDiag(B);
   
   // members of 'A'
   fsls_BSRMatrix *A = NULL;
   int     NNZ;
   int    *IA  = NULL;
   int    *JA  = NULL;
   double *val = NULL;
   
   // local variables
   int i,j,k,m;
   int nc2 = nc*nc;
   int ngridplus1 = ngrid + 1;
   
   // compute NNZ
   NNZ = ngrid;
   for (i = 0; i < nband; i ++)
   {
      NNZ += (ngrid - abs(offsets[i]));
   } 
   
   // Create and Initialize a BSRMatrix 'A'
   A = fsls_CTAlloc(fsls_BSRMatrix, 1);
   fsls_BSRMatrixROW(A) = ngrid;
   fsls_BSRMatrixCOL(A) = ngrid;
   fsls_BSRMatrixNNZ(A) = NNZ;    
   fsls_BSRMatrixNb(A)  = nc;
   fsls_BSRMatrixStorageManner(A) = 0;
   IA  = fsls_CTAlloc(int, ngridplus1);
   JA  = fsls_CTAlloc(int, NNZ);
   val = fsls_CTAlloc(double, NNZ*nc2); // bugged by Hu Xiaozhe
   fsls_BSRMatrixIA(A)  = IA;
   fsls_BSRMatrixJA(A)  = JA;
   fsls_BSRMatrixVal(A) = val;
   
   // Generate 'IA'
   for (i = 1; i < ngridplus1; i ++) IA[i] = 1; // take the diagonal blocks into account
   for (i = 0; i < nband; i ++)
   {
      k = offsets[i];
      if (k < 0)
      {
         for (j = -k+1; j < ngridplus1; j ++)
         {
            IA[j] ++;
         }
      }
      else
      {
         m = ngridplus1 - k;
         for (j = 1; j < m; j ++)
         {
            IA[j] ++;
         }
      }
   }
   IA[0] = 0;
   for (i = 1; i < ngridplus1; i ++)
   {
      IA[i] += IA[i-1];
   }
   
   // Generate 'JA' and 'val' at the same time
   for (i = 0 ; i < ngrid; i ++)
   {
      memcpy(val + IA[i]*nc2, diag + i*nc2, nc2*sizeof(double));
      JA[IA[i]] = i;
      IA[i] ++;
   }
   
   for (i = 0; i < nband; i ++)
   {
      k = offsets[i];
      if (k < 0)
      {
         for (j = -k; j < ngrid; j ++)
         {
            m = j + k;
            memcpy(val+IA[j]*nc2, offdiag[i]+m*nc2, nc2*sizeof(double));
            JA[IA[j]] = m;
            IA[j] ++;
         }
      }
      else
      {
         m = ngrid - k;
         for (j = 0; j < m; j ++)
         {
            memcpy(val + IA[j]*nc2, offdiag[i] + j*nc2, nc2*sizeof(double));
            //JA[IA[i]] = j - k;
            JA[IA[j]] = j + k;
            IA[j] ++;
         }
      }
   }

   // Map back the real 'IA' for BSR of A
   for (i = ngrid; i > 0; i --)
   {
      IA[i] = IA[i-1];
   }
   IA[0] = 0; 
   
   return (A);
}

/*!
 * \fn int fsls_STR2CSRMatrixDF
 * \brief Transfer a 'fsls_STRMatrix' type matrix into a 'fsls_CSRMatrix' type matrix.
 * \param fsls_STRMatrix *A pointer to the 'fsls_STRMatrix' type matrix
 * \param fsls_CSRMatrix **B_ptr pointer to the pointer of the 'fsls_CSRMatrix' type matrix
 * \note The diagonal submatrices are treated firstly. 
 * \author Yue Xiaoqiang (modified by peghoty)
 * \date 2010/04/29 
 */
int 
fsls_STR2CSRMatrixDF( fsls_STRMatrix *A, fsls_CSRMatrix **B_ptr )
{
   // some members of A
   int nc    = fsls_STRMatrixNc(A);   
   int ngrid = fsls_STRMatrixNgrid(A);
   int nband = fsls_STRMatrixNband(A);
   int     *offsets = fsls_STRMatrixOffSets(A);
   double  *diag    = fsls_STRMatrixDiag(A);
   double **offdiag = fsls_STRMatrixOffDiag(A);
   
   // some members of B
   int glo_row = nc*ngrid;
   int glo_nnz;
   int     *ia = NULL;
   int     *ja = NULL;
   double  *a  = NULL;
   
   fsls_CSRMatrix *B = NULL;

   // local variables
   int width;
   int nc2 = nc*nc;
   int BAND,ROW,COL;
   int ncb,nci;
   int row_start,col_start;
   int block; // how many blocks in the current ROW
   int i,j;
   int pos;
   int start;
   int val_L_start,val_R_start;
   int row;
   int tmp_col;
   double tmp_val;

   // allocate for 'ia' array
   ia = fsls_CTAlloc(int, glo_row+1);

   // Generate the 'ia' array
   ia[0] = 0;
   for (ROW = 0; ROW < ngrid; ROW ++)
   {
        block = 1; // diagonal block
        for (BAND = 0; BAND < nband; BAND ++)
        {
             width = offsets[BAND];
             COL   = ROW + width;
             if (width < 0)
             {
                 if (COL >= 0) block ++;
             }
             else
             {
                 if (COL < ngrid) block ++;
             }
        } // end for BAND
        
        ncb = nc*block;
        row_start = ROW*nc;
        
        for (i = 0; i < nc; i ++)
        {
            row = row_start + i; 
            ia[row+1] = ia[row] + ncb;
        }
   } // end for ROW

   // allocate for 'ja' and 'a' arrays
   glo_nnz = ia[glo_row];
   ja = fsls_CTAlloc(int, glo_nnz);
   a  = fsls_CTAlloc(double, glo_nnz);
    
   // Generate the 'ja' and 'a' arrays at the same time 
   for (ROW = 0; ROW < ngrid; ROW ++)
   {
        row_start = ROW*nc;    
        val_L_start = ROW*nc2;
		
        // deal with the diagonal band
        for (i = 0; i < nc; i ++)
        {
             nci   = nc*i;
             row   = row_start + i;
             start = ia[row];
             for (j = 0; j < nc; j ++)
             {
                  pos     = start + j;
                  ja[pos] = row_start + j;
                  a[pos]  = diag[val_L_start+nci+j];
             }
        }
        block = 1;
		
        // deal with the off-diagonal bands
        for (BAND = 0; BAND < nband; BAND ++)
        {
             width     = offsets[BAND]; 
             COL       = ROW + width;    
             ncb       = nc*block;
             col_start = COL*nc;
             
             if (width < 0)
             {
                  if (COL >= 0)
                  {
                       val_R_start = COL*nc2;
                       for (i = 0; i < nc; i ++)
                       {
                            nci = nc*i;
                            row = row_start + i;
                            start = ia[row];
                            for (j = 0 ; j < nc; j ++)
                            {
                                 pos     = start + ncb + j;
                                 ja[pos] = col_start + j;
                                 a[pos]  = offdiag[BAND][val_R_start+nci+j];
                            }
                       }
                       block ++;
                  }
             }
             else
             {
                  if (COL < ngrid)
                  {
                       for (i = 0; i < nc; i ++)
                       {
                            nci = nc*i;
                            row = row_start + i;
                            start = ia[row];
                            for (j = 0; j < nc; j ++)
                            {
                                 pos = start + ncb + j;
                                 ja[pos] = col_start + j;
                                 a[pos]  = offdiag[BAND][val_L_start+nci+j];
                            }
                       }
                       block ++;
                  }
             }
        }
   }

   // Reordering in such manner that every diagonal element 
   // is firstly stored in the corresponding row
   if (nc > 1)
   {
      for (ROW = 0; ROW < ngrid; ROW ++)
      {
         row_start = ROW*nc;
         for (j = 1; j < nc; j ++)
         {
            row   = row_start + j;
            start = ia[row];
            pos   = start + j;
            
            // swap in 'ja'
            tmp_col   = ja[start];
            ja[start] = ja[pos];
            ja[pos]   = tmp_col;
         
            // swap in 'a'
            tmp_val  = a[start];
            a[start] = a[pos];
            a[pos]   = tmp_val;            
         }
      }
   } 
   
   B = fsls_CSRMatrixCreate(glo_row, glo_row, glo_nnz);
   fsls_CSRMatrixI(B) = ia;
   fsls_CSRMatrixJ(B) = ja;
   fsls_CSRMatrixData(B) = a;
     
   *B_ptr = B;

   return (0);
}

/*!
 * \fn int fsls_STR2CSRMatrixDL
 * \brief Transfer a 'fsls_STRMatrix' type matrix into a 'fsls_CSRMatrix' type matrix.
 * \param fsls_STRMatrix *A pointer to the 'fsls_STRMatrix' type matrix
 * \param fsls_CSRMatrix **B_ptr pointer to the pointer of the 'fsls_CSRMatrix' type matrix
 * \note The diagonal submatrice are treated lastly.
 * \author Yue Xiaoqiang (modified by peghoty)
 * \date 2010/05/03 
 */
int 
fsls_STR2CSRMatrixDL(fsls_STRMatrix *A, fsls_CSRMatrix **B_ptr)
{
   // some members of A
   int nc    = fsls_STRMatrixNc(A);   
   int ngrid = fsls_STRMatrixNgrid(A);
   int nband = fsls_STRMatrixNband(A);
   int     *offsets = fsls_STRMatrixOffSets(A);
   double  *diag    = fsls_STRMatrixDiag(A);
   double **offdiag = fsls_STRMatrixOffDiag(A);
   
   // some members of B
   int glo_row = nc*ngrid;
   int glo_nnz;
   int     *ia = NULL;
   int     *ja = NULL;
   double  *a  = NULL;
   
   fsls_CSRMatrix *B = NULL;

   // local variables
   int width;
   int nc2 = nc*nc;
   int BAND,ROW,COL;
   int ncb,nci;
   int row_start,col_start;
   int block; // how many blocks in the current ROW
   int i,j;
   int pos;
   int start;
   int val_L_start,val_R_start;
   int row;

   // allocate for 'ia' array
   ia = fsls_CTAlloc(int, glo_row+1);

   // Generate the 'ia' array
   ia[0] = 0;
   for (ROW = 0; ROW < ngrid; ROW ++)
   {
        block = 1; // diagonal block
        for (BAND = 0; BAND < nband; BAND ++)
        {
             width = offsets[BAND];
             COL   = ROW + width;
             if (width < 0)
             {
                 if (COL >= 0) block ++;
             }
             else
             {
                 if (COL < ngrid) block ++;
             }
        } // end for BAND
        
        ncb = nc*block;
        row_start = ROW*nc;
        
        for (i = 0; i < nc; i ++)
        {
            row = row_start + i; 
            ia[row+1] = ia[row] + ncb;
        }
   } // end for ROW

   // allocate for 'ja' and 'a' arrays
   glo_nnz = ia[glo_row];
   ja = fsls_CTAlloc(int, glo_nnz);
   a  = fsls_CTAlloc(double, glo_nnz);
    
   // Generate the 'ja' and 'a' arrays at the same time 
   for (ROW = 0; ROW < ngrid; ROW ++)
   {
        row_start = ROW*nc;    
        val_L_start = ROW*nc2;
		
        block = 0;
		
        // deal with the off-diagonal bands
        for (BAND = 0; BAND < nband; BAND ++)
        {
             width     = offsets[BAND]; 
             COL       = ROW + width;    
             ncb       = nc*block;
             col_start = COL*nc;
             
             if (width < 0)
             {
                  if (COL >= 0)
                  {
                       val_R_start = COL*nc2;
                       for (i = 0; i < nc; i ++)
                       {
                            nci = nc*i;
                            row = row_start + i;
                            start = ia[row];
                            for (j = 0 ; j < nc; j ++)
                            {
                                 pos     = start + ncb + j;
                                 ja[pos] = col_start + j;
                                 a[pos]  = offdiag[BAND][val_R_start+nci+j];
                            }
                       }
                       block ++;
                  }
             }
             else
             {
                  if (COL < ngrid)
                  {
                       for (i = 0; i < nc; i ++)
                       {
                            nci = nc*i;
                            row = row_start + i;
                            start = ia[row];
                            for (j = 0; j < nc; j ++)
                            {
                                 pos = start + ncb + j;
                                 ja[pos] = col_start + j;
                                 a[pos]  = offdiag[BAND][val_L_start+nci+j];
                            }
                       }
                       block ++;
                  }
             }
        }
        
        // deal with the diagonal band
        ncb = nc*block;
        
        for (i = 0; i < nc; i ++)
        {
             nci   = nc*i;
             row   = row_start + i;
             start = ia[row];
             for (j = 0; j < nc; j ++)
             {
                  pos     = start + ncb + j;
                  ja[pos] = row_start + j;
                  a[pos]  = diag[val_L_start+nci+j];
             }
        }          
   }
   
   B = fsls_CSRMatrixCreate(glo_row, glo_row, glo_nnz);
   fsls_CSRMatrixI(B) = ia;
   fsls_CSRMatrixJ(B) = ja;
   fsls_CSRMatrixData(B) = a;
     
   *B_ptr = B;

   return (0);
}

/*!
 * \fn int fsls_STR2FullMatrix
 * \brief Transfer a 'fsls_STRMatrix' type matrix into a full matrix.
 * \param fsls_STRMatrix *A pointer to the 'fsls_STRMatrix' type matrix
 * \param double **full_ptr pointer to full matrix
 * \date 2010/05/03
 * \author Yue Xiaoqiang (modified by peghoty)
 */
int 
fsls_STR2FullMatrix( fsls_STRMatrix *A, double **full_ptr )
{
   // some members of A
   int nc    = fsls_STRMatrixNc(A);   
   int ngrid = fsls_STRMatrixNgrid(A);
   int nband = fsls_STRMatrixNband(A);
   int     *offsets = fsls_STRMatrixOffSets(A);
   double  *diag    = fsls_STRMatrixDiag(A);
   double **offdiag = fsls_STRMatrixOffDiag(A);
    
   int i,j,k,l;
   int nc2 = nc*nc;
   int start_diag;
   int start_row;
   int width;        
   int column;
   int start_f,start_d;
   int size = ngrid*nc;
   
   double *full = NULL;
 
   full = fsls_CTAlloc(double, size*size);    
    
   for (i = 0; i < ngrid; i ++)
   {
        // deal with the diagonal band
        start_diag = i*nc2;
        start_row = i*nc;
        for (j = 0; j < nc; j ++)
        {   
            start_f = (start_row + j)*size + start_row;
            start_d = start_diag + j*nc;
            for (k = 0; k < nc; k ++)
            {
                full[start_f+k] = diag[start_d+k];
            }
        }
        
        // deal with the off-diagonal bands
        for (l = 0; l < nband; l ++)
        {
            width  = offsets[l];
            column = i + width;
            if (width < 0)
            {
               if (column >= 0)
               {
                   for (j = 0; j < nc; j ++)
                   {
                       start_f = (start_row + j)*size + column*nc;
                       start_d = column*nc2 + j*nc;
                       for (k = 0; k < nc; k ++)
                       {
                           full[start_f+k] = offdiag[l][start_d+k];
                       }
                   }
               }
            }
            else
            {
               if (column < ngrid)
               {
                   for (j = 0; j < nc; j ++)
                   {
                       start_f = (start_row + j)*size + column*nc;
                       start_d = start_diag + j*nc;
                       for (k = 0; k < nc; k ++)
                       {
                           full[start_f+k] = offdiag[l][start_d+k];
                       }
                    }
               }
            }
        }
   }
   
   *full_ptr = full;
   
   return 0;
}

/*!
 * \fn int fsls_STR2XSTRMatrix
 * \brief Transfer a 'fsls_STRMatrix' type matrix into a 'fsls_XSTRMatrix' type matrix.
 * \param fsls_STRMatrix *A pointer to the 'fsls_STRMatrix' type matrix
 * \param fsls_XSTRMatrix **B_ptr pointer to the pointer to a 'fsls_XSTRMatrix' type matrix
 * \date 2010/05/06
 * \author Yue Xiaoqiang (modified by peghoty)
 */
int 
fsls_STR2XSTRMatrix( fsls_STRMatrix *A, fsls_XSTRMatrix **B_ptr )
{
    // some members of A
    int nc    = fsls_STRMatrixNc(A);   
    int ngrid = fsls_STRMatrixNgrid(A);
    int nband = fsls_STRMatrixNband(A);
    int     *offsets = fsls_STRMatrixOffSets(A);
    double  *diag    = fsls_STRMatrixDiag(A);
    double **offdiag = fsls_STRMatrixOffDiag(A);

    int       xdiag_index = -1;   // the index of the diagonal band
    int      *xoffsets = NULL;    // offsets of the bands (length is nband)
    double  **xdata    = NULL;    // matrix entries in all the bands
    int     **xcolumn  = NULL;    // column numbers of matrix entries
    
    fsls_XSTRMatrix *B = NULL;    

    int   i,ii,jj;
    int   pp,qq,uu,vv,ww;
    int   start_col;
    int   row,band,blk,width;
    int   rowleft;
    int  *bandmapn2o = NULL;  // band number mapping from fsls_XSTRMatrix to fsls_STRMatrix
    
    int   xnband   = nband + 1;
    int   nc2      = nc*nc;
    int   dis      = nc*xnband;
    int   numcol   = xnband*nc2;
    int   datasize = ngrid*numcol;  
    
    int   flag = 0;  // Compute the xcolumn(flag=1) or not(flag=0)
    
    B = fsls_CTAlloc(fsls_XSTRMatrix, 1);

    xoffsets = (int *)malloc(sizeof(int)*xnband);
    bandmapn2o = (int *)malloc(sizeof(int)*xnband);

    for (i = 0; i < nband; i ++)
    {
       xoffsets[i] = offsets[i];
    }
    xoffsets[nband] = 0; // this corresbands to the diagonal band
    
    for (i = 0; i < xnband; i ++)
    {
       bandmapn2o[i] = i;
    }

    // reorder 'bandmapn2o' and 'xoffsets' ascendingly
    fsls_iiQuickSort12(bandmapn2o, xoffsets, 0, nband);

#if 0 
    for(i=0; i<xnband; i++)
    {
       printf(" xoffsets[%d] = %d\n", i, xoffsets[i]);
    }
    for(i=0; i<xnband; i++)
    {
       printf(" bandmapn2o[%d] = %d\n", i, bandmapn2o[i]);
    }    
#endif

    if (flag) 
    {
       xcolumn = (int **)malloc(sizeof(int *)*ngrid);
       xcolumn[0] = (int *)malloc(datasize*sizeof(int));
    }
    xdata = (double **)malloc(sizeof(double *)*ngrid);
    xdata[0] = (double *)malloc(datasize*sizeof(double));
    memset(xdata[0], 0x0, datasize*sizeof(double));

    for (i = 1; i < ngrid; i ++)
    {
        if (flag) xcolumn[i] = xcolumn[0] + i*numcol;
        xdata[i] = xdata[0] + i*numcol;
    }


    // row->row; j->column; band->band; blk->submatrix
    for (band = 0; band < xnband; band ++)
    {
        width = xoffsets[band];
        start_col = band*nc;

        if (width < 0)
        {
            for (row = -width, blk = 0; row < ngrid; row ++, blk ++)
            {
                uu = blk*nc;
                vv = blk*nc2;
                for (ii = 0; ii < nc; ii ++)
                {
                    pp = start_col + ii*dis;
                    qq = vv + ii*nc;
                    for (jj = 0; jj < nc; jj ++)
                    {
                        xdata[row][pp+jj] = offdiag[bandmapn2o[band]][qq+jj];
                        if (flag) xcolumn[row][pp+jj] = uu + jj;
                    }
                }
            }
        }
        else if (width == 0)
        {
            xdiag_index = band;
            for (row = 0; row < ngrid; row ++)
            {   
                uu = row*nc;
                vv = row*nc2;
                for (ii = 0; ii < nc; ii ++)
                {
                    ww = ii*nc;
                    pp = start_col + ii*dis;
                    for (jj = 0; jj < nc; jj ++)
                    {
                        xdata[row][pp+jj] = diag[vv+ww+jj];
                        if (flag) xcolumn[row][pp+jj] = uu + jj;
                    }
                }
            }
        }
        else
        {
            rowleft = ngrid - width;
            for (row = 0; row < rowleft; row ++)
            {
                qq = (row+width)*nc;
                vv = row*nc2;
                for (ii = 0; ii < nc; ii ++)
                {
                    pp = start_col + ii*dis;
                    uu = ii*nc;
                    for (jj = 0; jj < nc; jj ++)
                    {
                        xdata[row][pp+jj] = offdiag[bandmapn2o[band]][vv+uu+jj];
                        if (flag) xcolumn[row][pp+jj] = qq + jj;
                    }
                }
            }
        }
    }

    // fill the fsls
    fsls_XSTRMatrixNx(B)  = fsls_STRMatrixNx(A);
    fsls_XSTRMatrixNy(B)  = fsls_STRMatrixNy(A);
    fsls_XSTRMatrixNz(B)  = fsls_STRMatrixNz(A);
    fsls_XSTRMatrixNxy(B) = fsls_STRMatrixNxy(A);
    fsls_XSTRMatrixNgrid(B) = ngrid;
    fsls_XSTRMatrixNc(B)    = nc;
    fsls_XSTRMatrixNband(B) = xnband;
    if (xdiag_index < 0)
    printf("\n\n Warning: no diagonal band!\n\n");
    fsls_XSTRMatrixDiagIndex(B) = xdiag_index;
    fsls_XSTRMatrixOffSets(B)   = xoffsets;
    fsls_XSTRMatrixData(B)      = xdata;
    if (flag) 
      fsls_XSTRMatrixClumn(B) = xcolumn; 
    else
      fsls_XSTRMatrixClumn(B) = NULL;   

    fsls_TFree(bandmapn2o);
    
    *B_ptr = B;
    
    return 0;
}

/*!
 * \fn int fsls_XVector2Vector
 * \brief Transfer a 'fsls_XVector' type matrix into a 'fsls_Vector'.
 * \param fsls_XVector *x pointer to the 'fsls_XVector' type matrix
 * \param fsls_Vector **y_ptr pointer to the pointer to fsls_Vector
 * \author peghoty
 * \date 2010/09/14 
 */
int 
fsls_XVector2Vector( fsls_XVector *x, fsls_Vector **y_ptr )
{
   fsls_Vector *y = NULL;
   int          n = fsls_XVectorSize(x);
   double      *x_data = fsls_XVectorData(x);
   double      *y_data = NULL;
   
   y = fsls_SeqVectorCreate(n);
   fsls_SeqVectorInitialize(y);
   y_data = fsls_VectorData(y);
   
   fsls_ArrayCopy(n, x_data, y_data);

   *y_ptr = y;
   
   return 0;
}

