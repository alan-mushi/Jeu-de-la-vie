#define EMPTY 0
#define RED 1
#define BLUE 2

__global__ void init_kernel(int * domain, int domain_x)
{
	// Dummy initialization
	domain[blockIdx.y * domain_x + blockIdx.x * blockDim.x + threadIdx.x]
		= ((blockIdx.x+threadIdx.x) == 0 ? 1 : 0);
		//= (1664525ul * (blockIdx.x + threadIdx.y + threadIdx.x) + 1013904223ul) % 3;
}

// Reads a cell at (x+dx, y+dy)
__device__ int read_cell(int * source_domain, int x, int y, int dx, int dy, unsigned int domain_x, unsigned int domain_y)
{
    x = (unsigned int)(x + dx) % domain_x;	// Wrap around
    y = (unsigned int)(y + dy) % domain_y;
    return source_domain[y * domain_x + x];
}

__device__ void inc_color(int cell, int *nb_blue, int *nb_red) {
	if (cell == BLUE) (*nb_blue)++;
	else if (cell == RED) (*nb_red)++;
}

// Compute kernel
__global__ void life_kernel(int * source_domain, int * dest_domain, int domain_x, int domain_y)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
	int pos_x = threadIdx.x + 1, pos_y = threadIdx.y + 1;
	__shared__ int shared_bloc[18][10];

	// Read cell
    shared_bloc[pos_x][pos_y] = read_cell(source_domain, tx, ty, 0, 0, domain_x, domain_y);

	if (pos_x == 1) 
		shared_bloc[pos_x-1][pos_y] = read_cell(source_domain, tx, ty, -1, 0, domain_x, domain_y);
	else if (pos_x == 16) 
		shared_bloc[pos_x+1][pos_y] = read_cell(source_domain, tx, ty, 1, 0, domain_x, domain_y);

	if (pos_y == 1) 
		shared_bloc[pos_x][pos_y-1] = read_cell(source_domain, tx, ty, 0, -1, domain_x, domain_y);
	else if (pos_y == 8) 
		shared_bloc[pos_x][pos_y+1] = read_cell(source_domain, tx, ty, 0, 1, domain_x, domain_y);

	if (pos_x == 1 && pos_y == 1)
		shared_bloc[pos_x-1][pos_y-1] = read_cell(source_domain, tx, ty, -1, -1, domain_x, domain_y);
	else if (pos_x == 1 && pos_y == 8)
		shared_bloc[pos_x-1][pos_y+1] = read_cell(source_domain, tx, ty, -1, 1, domain_x, domain_y);
	else if (pos_x == 16 && pos_y == 1)
		shared_bloc[pos_x+1][pos_y-1] = read_cell(source_domain, tx, ty, 1, -1, domain_x, domain_y);
	else if (pos_x == 16 && pos_y == 8)
		shared_bloc[pos_x+1][pos_y+1] = read_cell(source_domain, tx, ty, 1, 1, domain_x, domain_y);
    
    // Read the 8 neighbors and count number of blue and red
	int nb_blue = 0, nb_red = 0;

	__syncthreads();

	inc_color(shared_bloc[pos_x][pos_y-1], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x][pos_y+1], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x-1][pos_y], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x+1][pos_y], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x+1][pos_y-1], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x-1][pos_y+1], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x-1][pos_y-1], &nb_blue, &nb_red);
	inc_color(shared_bloc[pos_x+1][pos_y+1], &nb_blue, &nb_red);

	// Compute new value
	int res = 0;

	if (shared_bloc[pos_x][pos_y] != 0 && (nb_blue + nb_red == 2 || nb_blue + nb_red == 3))
		res = shared_bloc[pos_x][pos_y];
	else if (shared_bloc[pos_x][pos_y] == 0 && nb_blue + nb_red == 3) {
		if (nb_blue > nb_red)
			res = BLUE;
		else
			res = RED;
	}
	
	// Write it in dest_domain
	dest_domain[ty * domain_x + tx] = res;
}