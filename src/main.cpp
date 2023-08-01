#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <barnes_hut.h>
#include <mpi.h>
#include <stdio.h>

using namespace std;

/*
STRUCT HELPER FUNCTIONS
*/

string body_to_string(struct body b)
{
    return to_string(b.index) + " " + to_string(b.x_pos) + " " + to_string(b.y_pos) + " " + to_string(b.mass) + " " + to_string(b.x_vel) + " " + to_string(b.y_vel);
}

/*
MAIN HELPER FUNCTIONS
*/

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-i inputfilename (char *): input file name" << std::endl;
        std::cout << "\t-o outputfilename (char *): output file name" << std::endl;
        std::cout << "\t-s steps (int): number of iterations" << std::endl;
        std::cout << "\t-t \theta(double): threshold for MAC" << std::endl;
        std::cout << "\t-d dt(double): timestep" << std::endl;
        std::cout << "\t-V: (OPTIONAL, see below) flag to turn on visualization window" << std::endl;
        exit(0);
    }

    opts -> visualization = false;

    struct option l_opts[] = {
        {"inputfilename", required_argument, NULL, 'i'},
        {"outputfilename", required_argument, NULL, 'o'},
        {"steps", required_argument, NULL, 's'},
        {"theta", required_argument, NULL, 't'},
        {"dt", required_argument, NULL, 'd'},
        {"VISUALIZATION", optional_argument, NULL, 'V'}
    };

    int ind, c;
    //while ((c = getopt_long(argc, argv, "i:o:n:p:l:", l_opts, &ind)) != -1)
    while ((c = getopt_long(argc, argv, "i:o:s:t:Vd:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            opts->in_file = (char *) optarg;
            break;
        case 'o':
            opts->out_file = (char *) optarg;
            break;
        case 's':
            opts->steps = atoi((char *) optarg);
            break;
        case 't':
            opts->theta = atoi((char *)optarg);
            break;
        case 'd':
            opts->dt = atof((char *)optarg);
            break;
        case 'V':
            opts->visualization = true;
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}

void read_file(struct options_t* args,
               int*              n_bodies,
               struct body**     input_vals) {

	fstream in;

	in.open(args->in_file, ios::in);
	if (!in) {
		cout << "No such file";
	}
	else {
        // Get num vals
        in >> *n_bodies;

        // Alloc input and output arrays
        *input_vals = (struct body*) malloc(*n_bodies * sizeof(struct body));

        // Read input vals
        for (int i = 0; i < *n_bodies; ++i) {
            in >> (*input_vals)[i].index >> (*input_vals)[i].x_pos >> (*input_vals)[i].y_pos >> (*input_vals)[i].mass >> (*input_vals)[i].x_vel >> (*input_vals)[i].y_vel;
        }

	}
}

void write_file(struct options_t*         args,
                int                       num_bodies,
               	struct body*              bodies) {
    // Open file
	std::ofstream out;
	out.open(args->out_file, std::ofstream::trunc);

    out << num_bodies << endl;

	// Write solution to output file
	for (int i = 0; i < num_bodies; ++i) {
		out << body_to_string(bodies[i]) << endl;
	}

	out.flush();
	out.close();
}


int main(int argc, char **argv)
{
    // MPI Init
    int rank, size;
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank ); 
    MPI_Comm_size( MPI_COMM_WORLD, &size ); 

    // MPI Custom Data Type
	// Body
    const int body_nitems = 6;
    int body_blocklengths[6] = {1,1,1,1,1,1};
    MPI_Datatype body_types[6] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Datatype mpi_body_struct;
    MPI_Aint offsets[6];

    offsets[0] = offsetof(body, index);
    offsets[1] = offsetof(body, x_pos);
    offsets[2] = offsetof(body, y_pos);
    offsets[3] = offsetof(body, mass);
    offsets[4] = offsetof(body, x_vel);
    offsets[5] = offsetof(body, y_vel);

    MPI_Type_create_struct(body_nitems, body_blocklengths, offsets, body_types, &mpi_body_struct);
    MPI_Type_commit(&mpi_body_struct);

    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    // Graphics
    GLFWwindow* window;
    if (opts.visualization) {
        if (rank == 0) {
            window = init_vis();
            if (window == NULL)
                cout << "Graphics Error on INIT!" << endl;
        }
    }
    
    // Read in data points
    int num_bodies;
    struct body *bodies, *bodies_nxt;
    if (rank == 0) {
        read_file(&opts, &num_bodies, &bodies);
    }
    
    MPI_Bcast(&num_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    if (rank != 0) {
        bodies = (struct body*) malloc(num_bodies * sizeof(struct body));
    }

    MPI_Bcast(bodies, num_bodies, mpi_body_struct, 0, MPI_COMM_WORLD); 

    double start = MPI_Wtime();

    // Barnes Hut Algorithm
    for (int s=0; s<opts.steps; s++) {
        // Quad Tree Initialization
        struct quad_tree_node root;
        zero_quad_tree(&root, 0, MAX_COORD_VAL, 0, MAX_COORD_VAL);

        for (int p=0; p<num_bodies; p++) {
            insert_particle(&root, bodies[p]);
        }

		if (opts.visualization && rank == 0) {
            refresh_and_draw(num_bodies, bodies, &root, window);
		}

        // Force Calculation
        bodies_nxt = (struct body*) malloc(num_bodies * sizeof(struct body));

		int rank_elem_sz = num_bodies / size;
		if (rank == size - 1) rank_elem_sz += num_bodies % size;
        struct body *rank_bodies = (struct body *) malloc(sizeof(struct body) * rank_elem_sz);
		int dis_st = 0;
		int* scounts = (int*) malloc(size*sizeof(int)); 
    	int* displs = (int*) malloc(size*sizeof(int));
        for (int p=0; p<size; p++) {
            scounts[p] = num_bodies / size;
            displs[p] = dis_st;
            if (p == size - 1) {
                scounts[p] += num_bodies % size;
            }
            dis_st += scounts[p];
        }

        MPI_Scatterv(bodies, scounts, displs, mpi_body_struct, rank_bodies, rank_elem_sz, mpi_body_struct, 0, MPI_COMM_WORLD);


        for (int p=0; p<rank_elem_sz; p++) {
            double st = MPI_Wtime();
            double Fx = 0;
            double Fy = 0;

            calculate_force(&root, rank_bodies[p], &Fx, &Fy, opts.theta);
            recompute_position(&(rank_bodies[p]), Fx, Fy, opts.dt); 
            double en = MPI_Wtime();
		}

		MPI_Gatherv(rank_bodies, rank_elem_sz, mpi_body_struct, bodies_nxt, scounts, displs, mpi_body_struct, 0, MPI_COMM_WORLD);

        MPI_Bcast(bodies_nxt, num_bodies, mpi_body_struct, 0, MPI_COMM_WORLD); 

        free(rank_bodies);
        free(bodies);
        bodies = bodies_nxt;
    }
    
    double end = MPI_Wtime();
    
    if (rank == 0) {
        cout << end - start << endl;
        // Print output
        write_file(&opts, num_bodies, bodies);
    }

    // Free Input Data
    free(bodies);
    MPI_Type_free(&mpi_body_struct);

    MPI_Finalize();
}
