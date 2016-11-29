#include "../thnets.h"

int nnload_MulConstant(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_MulConstant;
	mod->updateOutput = nn_MulConstant_updateOutput;
	struct MulConstant *m = &mod->MulConstant;
	m->constant_scalar = TableGetNumber(t, "constant_scalar");
	m->inplace = TableGetBoolean(t, "inplace");
	return 0;
}

THFloatTensor *nn_MulConstant_updateOutput(struct module *module, THFloatTensor *input)
{
	float constant_scalar = module->MulConstant.constant_scalar;
	THFloatTensor *output = module->output;
	int inPlace = module->MulConstant.inplace == 1;

	long i, n = THFloatTensor_nElement(input);
	if (inPlace)
	{
		for(i = 0; i < n; i++)
			input->storage->data[i] *= constant_scalar;
		THFloatTensor_set(output, input);
	} else {
		THFloatTensor_resizeAs(output, input);
		for(i = 0; i < n; i++)
			output->storage->data[i] = input->storage->data[i] * constant_scalar;
	}
	return output;
}
