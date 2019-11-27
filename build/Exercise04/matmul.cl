#define TILE_WIDTH 16

__kernel void _mmul(const int taille,
   __global float* d_A,
   __global float* d_B,
   __global float* d_C)
{
  // Récupère le positionnement du thread dans l'espace de travail
  int tx = get_global_id(0);
  int ty = get_global_id(1);

   // sp mémorise la valeur courante du produit scalaire
   // réalisée par le thread
   float sp = 0;
   for (int k = 0; k < taille; ++k)
   {
      float elementA = d_A[ty * taille + k];
      float elementB = d_B[k * taille + tx];
      sp += elementA * elementB;
   }

   // Chaque thread écrit le résultat final en mémoire globale
   d_C[ty * taille + tx] = sp;
}

__kernel void mmul(const int taille,
    __global float* d_A,
    __global float* d_B,
    __global float* d_C)
{
  // Alloue la mémoire partagée pour récupération parallèle des données
  __local float ds_M[TILE_WIDTH][TILE_WIDTH];
  __local float ds_N[TILE_WIDTH][TILE_WIDTH];

  // Récupère le position local du work-item et du groupe de travail
  int bx = get_group_id(0); int by = get_group_id(1);
  int tx = get_local_id(0); int ty = get_local_id(1);

  // Position de l'element de P sur lequel on travail
  int Col = bx * TILE_WIDTH + tx;
  int Row = by * TILE_WIDTH + ty;
  float sp = 0;

  // Boucle sur l'ensemble les blocs de M et N necessaire pour
  // calculer un element de
  for (int m = 0; m < taille/TILE_WIDTH; ++m) {
    // Chargement collaboratif en memoire partagee
    ds_M[ty][tx] = d_A[Row*taille + m*TILE_WIDTH+tx];
    ds_N[ty][tx] = d_B[(m*TILE_WIDTH+ty)*taille+Col];

    // Force la synchronisation à l'intérieur d'un groupe pour assurer
    // l'exactitude des calculs
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TILE_WIDTH; ++k)
      sp += ds_M[ty][k] * ds_N[k][tx];

    // Force la synchronisation pour assurer que les calculs sont bien terminés
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Chaque thread écrit le résultat final en mémoire globale
  d_C[Row*taille+Col] = sp;
}
