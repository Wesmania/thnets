#include <math.h>
#include "../thnets.h"

int nnload_Sqrt(struct module *mod, struct nnmodule *n)
{
	/* there is an additional 'eps' argument that apparently does nothing */
	mod->type = MT_Sqrt;
	mod->updateOutput = nn_Sqrt_updateOutput;
	return 0;
}

THFloatTensor *nn_Sqrt_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THFloatTensor_resize(output, input->size, input->nDimension);

	float *output_data = THFloatTensor_data(output);
	float *input_data = THFloatTensor_data(input);

	long l, nelem = THFloatTensor_nElement(input);

#pragma omp parallel for private(l)
	for (l = 0; l < nelem; l++)
		output_data[l] = pow(input_data[l], 0.5);

	return output;
}
