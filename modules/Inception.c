#include <math.h>
#include <string.h>
#include "../thnets.h"

int nnload_Inception(struct module *mod, struct nnmodule *n)
{
	nnload_DepthConcat(mod, n);
	mod->type = MT_Inception;
	struct network *net = Module2Network(n);
	mod->Inception.nelem = net->nelem;
	mod->Inception.modules = net->modules;
	free(net);
	mod->updateOutput = nn_Inception_updateOutput;
	return 0;
}

THFloatTensor *nn_Inception_updateOutput(struct module *module, THFloatTensor *input)
{
	int batchdim = input->nDimension + 1;
	int batched = 0;
	THFloatTensor *output = module->output;
	long newdims[batchdim];
	struct module *modules = module->Inception.modules;

	/* batch */
	if (input->nDimension == 3) {
		memcpy(newdims + 1, input->size, input->nDimension * sizeof(*input->size));
		newdims[0] = 1;
		THFloatTensor_resize(input, newdims, batchdim);
		batched = 1;
	}

	modules[0].updateOutput(&modules[0], input);
	THFloatTensor_set(output, modules[0].output);

	/* debatch */
	if (batched) {
		memcpy(newdims, output->size, batchdim * sizeof(*input->size));
		THFloatTensor_resize(output, newdims + 1, batchdim - 1);
	}

	return output;
}
