#include <unistd.h>
#include <mpi.h>

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

} // namespace dmmio::utils