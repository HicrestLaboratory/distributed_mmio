#include <unistd.h>
#include <mpi.h>
#include <string.h>

#include "../../include/dmmio/dmmio.h"

namespace dmmio::utils {

  void ProcessGrid_print(const dmmio::ProcessGrid *grid, FILE* fp) {
	  if (grid->global_rank==0) {
      fprintf(fp, "========================\n");
      fprintf(fp, " ProcessGrid Details \n");
      fprintf(fp, "========================\n");
      fprintf(fp, "Total processes:\t %d\n", grid->global_size);
      fprintf(fp, "row size:\t %d\n", grid->row_size);
      fprintf(fp, "col size:\t %d\n", grid->col_size);
      fprintf(fp, "node size:\t %d\n", grid->node_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
	  sleep(1);

    for (int i=0; i<grid->global_size; i++) {
      if (grid->global_rank == i) {
        fprintf(fp, "----- Process %d -----\n", grid->global_rank);
        fprintf(fp, "Rank:\t %d\n", grid->global_rank);
        fprintf(fp, "row rank:\t %d\n", grid->row_rank);
        fprintf(fp, "col rank:\t %d\n", grid->col_rank);
        fprintf(fp, "node rank:\t %d\n", grid->node_rank);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    fflush(stdout);
    sleep(1);
    if (grid->global_rank == 0) fprintf(fp, "========================\n");
    MPI_Barrier(MPI_COMM_WORLD);
  }

  typedef struct {
      int col_rank;
      int row_rank;
      int node_rank;

      int  hostname_len;
      char hostname[MPI_MAX_PROCESSOR_NAME];

      int gid;
  } nodeInfo;

  nodeInfo* genNodeInfo(const dmmio::ProcessGrid *grid) {
      nodeInfo *myinfo  = (nodeInfo*)malloc(sizeof(nodeInfo));
      myinfo->col_rank  = grid->col_rank;
      myinfo->row_rank  = grid->row_rank;
      myinfo->node_rank = grid->node_rank;

      MPI_Get_processor_name(myinfo->hostname, &(myinfo->hostname_len));
      return(myinfo);
  }

  void genEmptyNodeInfo(nodeInfo* emptyinfo) {
      emptyinfo->col_rank  = -1;
      emptyinfo->row_rank  = -1;
      emptyinfo->node_rank = -1;

      const char *msg = "Uninitialized";
      emptyinfo->hostname_len = strlen(msg);
      snprintf(emptyinfo->hostname, MPI_MAX_PROCESSOR_NAME, "%s", msg);
  }

  void overwriteNodeInfo (nodeInfo* destination, nodeInfo* source) {
      destination->col_rank   = source->col_rank;
      destination->row_rank   = source->row_rank;
      destination->node_rank  = source->node_rank;

      destination->hostname_len = source->hostname_len;
      memcpy(destination->hostname, source->hostname, sizeof(char)*MPI_MAX_PROCESSOR_NAME);
  }

  int ProcessGrid_graph(const dmmio::ProcessGrid *grid, FILE* fp, bool host_gpu_id_print) {
      int row_size    = grid->row_size;
      int col_size    = grid->col_size;
      int node_size   = grid->node_size;
      int global_size = grid->global_size;

      // Allocate 3D grid dynamically: [row][col][node]
      nodeInfo ***tmp_grid = (nodeInfo ***)malloc(col_size * sizeof(nodeInfo **));
      for (int row = 0; row < col_size; ++row) {
          tmp_grid[row] = (nodeInfo **)malloc(row_size * sizeof(nodeInfo *));
          for (int col = 0; col < row_size; ++col) {
              tmp_grid[row][col] = (nodeInfo *)malloc(node_size * sizeof(nodeInfo));
              // Initialize with -1 (or any invalid rank)
              for (int node = 0; node < node_size; ++node) {
                  genEmptyNodeInfo(&(tmp_grid[row][col][node]));
              }
          }
      }

      // Populate the grid
      /*
      int myvalues[3];
      myvalues[0] = grid->col_rank;
      myvalues[1] = grid->row_rank;
      myvalues[2] = grid->node_rank;
      int *allvalues = (int*)malloc(sizeof(int)*global_size*3);
      */
      nodeInfo *myinfo  = genNodeInfo(grid);
      nodeInfo *allinfo = (nodeInfo*)malloc(sizeof(nodeInfo)*global_size);
      MPI_Allgather(myinfo, sizeof(nodeInfo), MPI_BYTE, allinfo, sizeof(nodeInfo), MPI_BYTE, MPI_COMM_WORLD);
      for (int i = 0; i < global_size; i++) {
          // tmp_grid[allvalues[3*i]][allvalues[3*i+1]][allvalues[3*i+2]] = i;
          overwriteNodeInfo(&(tmp_grid[allinfo[i].col_rank][allinfo[i].row_rank][allinfo[i].node_rank]), &(allinfo[i]));
          tmp_grid[allinfo[i].col_rank][allinfo[i].row_rank][allinfo[i].node_rank].gid = i;
      }

      int max_hostnamelen = 0;
      MPI_Allreduce(&(myinfo->hostname_len), &max_hostnamelen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

      // Header row: column labels
      if(grid->global_rank==0) fprintf(fp, "         ");
      for (int col = 0; col < row_size; ++col) {
          if(grid->global_rank==0) fprintf(fp, "col %-2d         ", col);
      }
      if(grid->global_rank==0) fprintf(fp, "\n");

      // Top border
      if(grid->global_rank==0) fprintf(fp, "       ");
      for (int col = 0; col < row_size; ++col) {
          if(grid->global_rank==0) fprintf(fp, "------------------- ");
      }
      if(grid->global_rank==0) fprintf(fp, "\n");

      // For each row
      for (int row = 0; row < col_size; ++row) {
          for (int node = 0; node < node_size; ++node) {
              if (node == 0) {
                  if(grid->global_rank==0) fprintf(fp, "row %-2d |", row);
              } else {
                  if(grid->global_rank==0) fprintf(fp, "       |");
              }

              for (int col = 0; col < row_size; ++col) {
                  int   gid = tmp_grid[row][col][node].gid;
                  char* hostname = tmp_grid[row][col][node].hostname;
                  if(grid->global_rank==0) {
                      if (host_gpu_id_print) {
                        fprintf(fp, " Node%d [%-3d] [%12s]   |", node, gid, hostname);
                      } else {
                        fprintf(fp, " Node%d [%-3d]     |", node, gid);
                      }
                  }
              }
              if(grid->global_rank==0) fprintf(fp, "\n");
          }

          // Separator line
          if(grid->global_rank==0) fprintf(fp, "       ");
          for (int col = 0; col < row_size; ++col) {
              if(grid->global_rank==0) fprintf(fp, "------------------- ");
          }
          if(grid->global_rank==0) fprintf(fp, "\n");
      }

      // Cleanup memory
      for (int row = 0; row < col_size; ++row) {
          for (int col = 0; col < row_size; ++col) {
              free(tmp_grid[row][col]);
          }
          free(tmp_grid[row]);
      }
      free(tmp_grid);
      free(myinfo);

      return 0;
  }

} // namespace dmmio::utils
