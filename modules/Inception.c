#include <math.h>
#include <string.h>
#include "../thnets.h"

int nnload_Inception(struct module *mod, struct nnmodule *n)
{
	nnload_DepthConcat(mod, n);
	mod->type = MT_Inception;
	mod->updateOutput = nn_Inception_updateOutput;
	return 0;
}

THFloatTensor *nn_Inception_updateOutput(struct module *module, THFloatTensor *input)
{
	int batchdim = input->nDimension + 1;
	THFloatTensor *output;
	long newdims[batchdim];

	memcpy(newdims, input->size, batchdim * sizeof(*input->size));
	newdims[batchdim - 1] = 1;
	THFloatTensor_resize(input, newdims, batchdim);

	output = nn_DepthConcat_updateOutput(module, input);

	THFloatTensor_resize(output, output->size, output->nDimension - 1);
	return output;
}
