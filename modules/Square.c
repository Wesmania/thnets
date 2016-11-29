#include <math.h>
#include "../thnets.h"

int nnload_Square(struct module *mod, struct nnmodule *n)
{
	mod->type = MT_Square;
	mod->updateOutput = nn_Square_updateOutput;
	return 0;
}

THFloatTensor *nn_Square_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	THFloatTensor_resize(output, input->size, input->nDimension);

	float *output_data = THFloatTensor_data(output);
	float *input_data = THFloatTensor_data(input);

	long l, nelem = 1;
	int i;

	for (i = 0; i < input->nDimension ; i++)
		nelem *= input->size[i];

#pragma omp parallel for private(l)
	for (l = 0; l < nelem; l++)
		output_data[l] = input_data[l] * input_data[l];

	return output;
}
