#include <barnes_hut.h>

/*
struct quad_tree_node {
		double d;
		double x_min, x_max, y_min, y_max;

    int num_particles;
    double com_x_pos, com_y_pos;
    double mass;

    struct body particle;
    struct quad_tree_node *q[4];
};
*/

double distance_coord(double coord1, double coord2)
{
		return coord2 - coord1;
}

double distance(struct body b1, struct body b2)
{
		double dist = sqrt(pow(b1.x_pos - b2.x_pos, 2) + pow(b1.y_pos - b2.y_pos, 2));
		return (dist < RLIMIT) ? RLIMIT : dist;
}

double distance(struct body b1, struct quad_tree_node *node)
{
		double dist = sqrt(pow(node->com_x_pos - b1.x_pos, 2) + pow(node->com_y_pos - b1.y_pos, 2));
		return (dist < RLIMIT) ? RLIMIT : dist;
}

int get_particle_quadrant(struct body particle, struct quad_tree_node *node)
{
		double mid_x = (node -> x_max + node -> x_min) / 2;
		double mid_y = (node -> y_max + node -> y_min) / 2;

		if (particle.x_pos >= node->x_min && particle.x_pos <= mid_x && particle.y_pos >= node->y_min && particle.y_pos <= mid_y)
				return 0;
		if (particle.x_pos > mid_x && particle.x_pos <= node->x_max && particle.y_pos >= node->y_min && particle.y_pos <= mid_y)
				return 1;
		if (particle.x_pos >= node->x_min && particle.x_pos <= mid_x && particle.y_pos > mid_y && particle.y_pos <= node->y_max)
				return 2;
		if (particle.x_pos > mid_x && particle.x_pos <= node->x_max && particle.y_pos > mid_y && particle.y_pos <= node->y_max)
				return 3;

		std::cout << "Error: Not suitable quadrant for particle found!" << std::endl;
		exit(1);
}

void zero_quad_tree(struct quad_tree_node *node, double min_x, double max_x, double min_y, double max_y)
{
		node -> d = max_x - min_x;
		node -> x_min = min_x;
		node -> x_max = max_x;
		node -> y_min = min_y;
		node -> y_max = max_y;
		node -> num_particles = 0;
}

void insert_particle(struct quad_tree_node *node, struct body particle)
{
		if (particle.mass == -1)
				return;

		if (node -> num_particles == 0) {						// Case 1: Zero Quadrant
				node -> num_particles = 1;
				node -> mass = particle.mass;
				node -> com_x_pos = particle.x_pos;
				node -> com_y_pos = particle.y_pos;
				node -> particle = particle;
		} else if (node -> num_particles == 1) {		// Case 2: Single Body Quadrant
				node -> num_particles = 2;

				// New Mass and COM calculations
				node -> com_x_pos = (node -> com_x_pos * node -> mass) + particle.x_pos * particle.mass;
				node -> com_y_pos = (node -> com_y_pos * node -> mass) + particle.y_pos * particle.mass;
				node -> mass += particle.mass;
				node -> com_x_pos /= node -> mass;
				node -> com_y_pos /= node -> mass;

				// Create Quadrants
				double mid_x = (node -> x_max + node -> x_min) / 2;
				double mid_y = (node -> y_max + node -> y_min) / 2;

				node -> q[0] = (struct quad_tree_node*) malloc(sizeof(struct quad_tree_node));
				zero_quad_tree(node->q[0], node->x_min, mid_x, node->y_min, mid_y);
				node -> q[1] = (struct quad_tree_node*) malloc(sizeof(struct quad_tree_node));
				zero_quad_tree(node->q[1], mid_x, node->x_max, node->y_min, mid_y);
				node -> q[2] = (struct quad_tree_node*) malloc(sizeof(struct quad_tree_node));
				zero_quad_tree(node->q[2], node->x_min, mid_x, mid_y, node->y_max);
				node -> q[3] = (struct quad_tree_node*) malloc(sizeof(struct quad_tree_node));
				zero_quad_tree(node->q[3], mid_x, node->x_max, mid_y, node->y_max);

				// Reassign particles to respective quadrants
				int old_particle_quad = get_particle_quadrant(node->particle, node);
				insert_particle(node->q[old_particle_quad], node->particle);

				int new_particle_quad = get_particle_quadrant(particle, node);
				insert_particle(node->q[new_particle_quad], particle);
		} else {																		// Case 3: Multi Body Quadrant
				node -> num_particles += 1;

				// New Mass and COM calculations
				node -> com_x_pos = (node -> com_x_pos * node -> mass) + particle.x_pos * particle.mass;
				node -> com_y_pos = (node -> com_y_pos * node -> mass) + particle.y_pos * particle.mass;
				node -> mass += particle.mass;
				node -> com_x_pos /= node -> mass;
				node -> com_y_pos /= node -> mass;

				// Find Next Quadrant to Insert
				int new_particle_quad = get_particle_quadrant(particle, node);
				insert_particle(node->q[new_particle_quad], particle);
		}
}

void calculate_force(struct quad_tree_node *node, struct body particle, double *Fx, double *Fy, double theta)
{
		if (particle.mass == -1) return;

		if (node -> num_particles == 0) return;

		if (node -> num_particles == 1) {						// Case 2: Single Body Quadrant
				*Fx += G * (node->particle).mass * particle.mass * distance_coord(particle.x_pos, (node->particle).x_pos) / pow(distance(node->particle, particle), 3);
				*Fy += G * (node->particle).mass * particle.mass * distance_coord(particle.y_pos, (node->particle).y_pos) / pow(distance(node->particle, particle), 3);
		} else {																		// Case 3: Multi Body Quadrant
				double r = distance(particle, node);

				if (node->d / r < theta) {
						*Fx += G * node->mass * particle.mass * distance_coord(particle.x_pos, node->com_x_pos) / pow(distance(particle, node), 3);
						*Fy += G * node->mass * particle.mass * distance_coord(particle.y_pos, node->com_y_pos) / pow(distance(particle, node), 3);
				} else {
						for (int i=0; i<4; i++) {
								calculate_force(node->q[i], particle, Fx, Fy, theta);
						}
				}
		}
}

void recompute_position(struct body *particle, double Fx, double Fy, double timestep)
{
		if (particle -> mass == -1) return;

		double ax = Fx / particle->mass;
		double ay = Fy / particle->mass;

		particle->x_pos += particle->x_vel*timestep + 0.5*ax*pow(timestep, 2);
		particle->y_pos += particle->y_vel*timestep + 0.5*ay*pow(timestep, 2);

		particle->x_vel += ax*timestep;
		particle->y_vel += ay*timestep;

		if (particle->x_pos < 0 || particle->x_pos > MAX_COORD_VAL || particle->y_pos < 0 || particle->y_pos > MAX_COORD_VAL) {
				particle->mass = -1;
		}

}