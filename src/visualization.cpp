#include <barnes_hut.h>

GLFWwindow* init_vis()
{
		/* OpenGL window dims */
		int width=600, height=600;
		GLFWwindow* window;
		if( !glfwInit() ){
				fprintf( stderr, "Failed to initialize GLFW\n" );
				return NULL;
		}
		// Open a window and create its OpenGL context
		window = glfwCreateWindow( width, height, "Simulation", NULL, NULL);
		if( window == NULL ){
				fprintf( stderr, "Failed to open GLFW window.\n" );
				glfwTerminate();
				return NULL;
		}
		glfwMakeContextCurrent(window); // Initialize GLEW
		if (glewInit() != GLEW_OK) {
				fprintf(stderr, "Failed to initialize GLEW\n");
				return NULL;
		}
		// Ensure we can capture the escape key being pressed below
		glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);	

		return window;	
}

double coordinate_transform(double coord)
{
		return 2*coord/MAX_COORD_VAL - 1;
}

void drawOctreeBounds2D(struct quad_tree_node *node) 
{
		int i;

		if(node == NULL || node -> num_particles <= 1)
				return;
		
		glBegin(GL_LINES);

		// set the color of lines to be white
		glColor3f(1.0f, 1.0f, 1.0f);
		
		// specify the start point's coordinates
		glVertex2f(coordinate_transform(node->x_min), coordinate_transform((node -> y_max + node -> y_min) / 2));
		
		// specify the end point's coordinates
		glVertex2f(coordinate_transform(node->x_max), coordinate_transform((node -> y_max + node -> y_min) / 2));
		
		// do the same for verticle line
		glVertex2f(coordinate_transform((node -> x_max + node -> x_min) / 2), coordinate_transform(node->y_min));
		glVertex2f(coordinate_transform((node -> x_max + node -> x_min) / 2), coordinate_transform(node->y_max));
		
		glEnd();
		
		for (int i=0; i<4; i++) {
				drawOctreeBounds2D(node->q[i]);
		}
}	

void drawParticle2D(double x_window, double y_window,
										double radius,
										double mass) 
{
    int k = 0;
    float angle = 0.0f;

    glBegin(GL_TRIANGLE_FAN);
    
		glColor3f(mass/MAX_COORD_VAL, 0.2f, 0.2f);
    
		glVertex2f(x_window, y_window);
    
		for(k=0;k<20;k++){
        angle=(float) (k)/19*2*3.141592;
        glVertex2f(x_window+radius*cos(angle),
           					y_window+radius*sin(angle));
    }
    
		glEnd();
}

void refresh_and_draw(int num_bodies, struct body* bodies, struct quad_tree_node *node, GLFWwindow* window)
{
		glClear( GL_COLOR_BUFFER_BIT );

		drawOctreeBounds2D(node);
		
		for(int p = 0; p < num_bodies; p++) {
				if (bodies[p].mass != -1)
						drawParticle2D(coordinate_transform(bodies[p].x_pos), coordinate_transform(bodies[p].y_pos), 0.01,bodies[p].mass);
		}
		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
}