#include <math.h>
#include <string.h>
#include "../thnets.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))

int nnload_DepthConcat(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_DepthConcat;
	mod->DepthConcat.dimension = TableGetNumber(t, "dimension") - 1;
	struct network *net = Module2Network(n);
	mod->DepthConcat.nelem = net->nelem;
	mod->DepthConcat.modules = net->modules;
	free(net);
	mod->updateOutput = nn_DepthConcat_updateOutput;
	return 0;
}

void copy_with_center(THFloatTensor *out, THFloatTensor *in, int offset, int depth_dim)
{
	int dim = out->nDimension;
	int loops[dim], offsets[dim];
	int lvl;
	long in_off = 0, out_off = 0;
	float *ints = THFloatTensor_data(in);
	float *outs = THFloatTensor_data(out);
	
	memset(loops, 0, dim * sizeof(*loops));
	for(lvl = 0; lvl < dim; lvl++)
	{
		if (lvl == depth_dim)
			offsets[lvl] = offset;
		else
			offsets[lvl] = ((out->size[lvl] - in->size[lvl]) / 2);

		out_off += offsets[lvl] * out->stride[lvl];
	}

	while(loops[0] < in->size[0]) {
		memcpy(outs + out_off, ints + in_off, in->size[dim - 1] * sizeof(*outs));
		loops[dim - 1] += in->size[dim - 1];
		in_off += in->stride[dim - 1] * in->size[dim - 1];
		out_off += out->stride[dim - 1] * in->size[dim - 1];
		lvl = dim - 1;
		while (lvl > 0 && loops[lvl] == in->size[lvl]) {
			loops[lvl] = 0;
			in_off -= in->stride[lvl] * in->size[lvl];
			out_off -= out->stride[lvl] * in->size[lvl];
			lvl--;
			loops[lvl]++;
			in_off += in->stride[lvl];
			out_off += out->stride[lvl];
		}
	}
}

THFloatTensor *nn_DepthConcat_updateOutput(struct module *module, THFloatTensor *input)
{
	THFloatTensor *output = module->output;
	int nelem = module->DepthConcat.nelem;
	int dimension = module->DepthConcat.dimension;
	int ndims;
	int i, j;
	struct module *modules = module->DepthConcat.modules;

	for(i = 0; i < nelem; i++)
		modules[i].updateOutput(&modules[i], input);
	ndims = modules[0].output->nDimension;

	// Check correctness
	for(i = 0; i < nelem; i++)
	{
		if(modules[i].output->nDimension != ndims)
			THError("Concatenation of tensors of different dimensionality");
	}

	long outputDims[ndims];
	memset(outputDims, 0, ndims * sizeof(*outputDims));

	for(i = 0; i < nelem; i++)
	{
		for(j = 0; j < ndims; j++)
		{
			if (j == dimension)
				outputDims[j] += modules[i].output->size[j];
			else
				outputDims[j] = MAX(outputDims[j], modules[i].output->size[j]);
		}
	}

	THFloatTensor_resize(output, outputDims, ndims);
	THFloatTensor_zero(output);

	long offset = 0;

	for(i = 0; i < nelem; i++) {
		copy_with_center(output, modules[i].output, offset, dimension);
		offset += modules[i].output->size[dimension];
	}

	return output;

}
