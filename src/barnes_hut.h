#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glu.h>
#include <GL/glut.h>


const double MAX_COORD_VAL = 4;
const double G = 0.0001;
const double RLIMIT = 0.03;

/*
-i inputfilename (char *): input file name
-o outputfilename (char *): output file name
-s steps (int): number of iterations
-t \theta(double): threshold for MAC
-d dt(double): timestep
-V: (OPTIONAL, see below) flag to turn on visualization window
*/

struct options_t {
    char *in_file;
    char *out_file;
    int steps;
    double theta;
    double dt;
    bool visualization;
};


struct body {
    int index;
    double x_pos, y_pos;
    double mass;
    double x_vel, y_vel;
};

struct quad_tree_node {
    double d;
    double x_min, x_max, y_min, y_max;

    int num_particles;
    double com_x_pos, com_y_pos;
    double mass;

    struct body particle;
    struct quad_tree_node *q[4];
};

/*
barnes_hut.cpp
*/
double distance_coord(double coord1, double coord2);
double distance(struct body b1, struct quad_tree_node *node);
double distance(struct body b1, struct body b2);
void zero_quad_tree(struct quad_tree_node *node, double min_x, double max_x, double min_y, double max_y);
void insert_particle(struct quad_tree_node *node, struct body particle);
int get_particle_quadrant(struct body particle, struct quad_tree_node *node);
void calculate_force(struct quad_tree_node *node, struct body particle, double *Fx, double *Fy, double theta);
void recompute_position(struct body *particle, double Fx, double Fy, double timestep);

/*
visualization.cpp
*/
GLFWwindow* init_vis();
double coordinate_transform(double coord);
void drawOctreeBounds2D(struct quad_tree_node *node);
void drawParticle2D(double x_window, double y_window, double radius, double mass);
void refresh_and_draw(int num_bodies, struct body* bodies, struct quad_tree_node *node, GLFWwindow* window);