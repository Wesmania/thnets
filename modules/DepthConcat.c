#include <math.h>
#include <string.h>
#include "../thnets.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))

int nnload_DepthConcat(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_DepthConcat;
	mod->DepthConcat.dimension = TableGetNumber(t, "dimension");
	struct network *net = Module2Network(n);
	mod->DepthConcat.nelem = net->nelem;
	mod->DepthConcat.modules = net->modules;
	free(net);
	mod->updateOutput = nn_DepthConcat_updateOutput;
	return 0;
}

void copy_with_center(THFloatTensor *out, THFloatTensor *in, int offset)
{
	int dim = out->nDimension;
	int loops[dim], offsets[dim];
	int lvl;
	long in_off = 0, out_off = 0;
	float *ints = THFloatTensor_data(in);
	float *outs = THFloatTensor_data(out);
	
	memset(loops, 0, dim);
	for(lvl = 0; lvl < dim - 1; lvl++)
	{
		offsets[lvl] = (out->size[lvl] - in->size[lvl] / 2);
		out_off += offsets[lvl] * out->stride[lvl];
	}

	while(loops[0] < in->size[0]) {
		lvl = dim - 2;
		loops[lvl]++;
		in_off += in->stride[lvl];
		out_off += out->stride[lvl];
		while (loops[lvl] == in->size[lvl]) {
			in_off -= in->stride[lvl] * in->size[lvl];
			out_off -= out->stride[lvl] * in->size[lvl];
			lvl--;
			loops[lvl]++;
			in_off += in->stride[lvl];
			out_off += out->stride[lvl];
		}
		memcpy(outs + out_off + offset, ints + in_off, in->stride[dim - 1] * sizeof(*ints));
	}
}

THFloatTensor *nn_DepthConcat_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	int nelem = module->DepthConcat.nelem;
	int dimension = module->DepthConcat.dimension;
	long outputDims[dimension];
	memset(outputDims, 0, dimension);

	int i, j;
	struct module *modules = module->DepthConcat.modules;
	for(i = 0; i < nelem; i++)
	{
		modules[i].updateOutput(&modules[i], input);
		outputDims[dimension - 1] += modules[i].output->size[dimension - 1];
	}
	// Check correctness
	for(i = 0; i < nelem; i++)
	{
		if(modules[i].output->nDimension != dimension)
			THError("Concatenation of tensors of different dimensionality");
	}

	for(i = 0; i < nelem; i++)
	{
		for(j = 0; j < dimension - 1; j++)
		{
			outputDims[j] = MAX(outputDims[j], modules[i].output->size[j]);
		}
	}

	THFloatTensor_resize(output, outputDims, dimension);
	THFloatTensor_zero(output);

	long offset = 0;

	for(i = 0; i < nelem; i++) {
		copy_with_center(output, modules[i].output, offset);
		offset += modules[i].output->size[dimension - 1];
	}

	return output;

}
