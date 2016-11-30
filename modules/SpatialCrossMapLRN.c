#include <math.h>
#include "../thnets.h"

int nnload_SpatialCrossMapLRN(struct module *mod, struct nnmodule *n)
{
	struct table *t = n->table;
	mod->type = MT_SpatialCrossMapLRN;
	mod->updateOutput = nn_SpatialCrossMapLRN_updateOutput;
	struct SpatialCrossMapLRN *m = &mod->SpatialCrossMapLRN;
	m->size = (int) (TableGetNumber(t, "size") + 0.5);
	m->alpha = TableGetNumber(t, "alpha");
	m->beta = TableGetNumber(t, "beta");
	m->k = TableGetNumber(t, "k");
	return 0;
}

THFloatTensor *nn_SpatialCrossMapLRN_updateOutput(struct module *module, THFloatTensor *input)
{
	int size = module->SpatialCrossMapLRN.size;
	float k = module->SpatialCrossMapLRN.k;
	float alpha = module->SpatialCrossMapLRN.alpha;
	float beta = module->SpatialCrossMapLRN.beta;
	THFloatTensor *output = module->output;

	float *output_data;
	float *input_data;
	float *scale_data;

	int dimw = 2;
	int dimh = 1;
	int dimc = 0;
	long nbatch = 1;

	long inputWidth;
	long inputHeight;
	long channels;

	if(! (input->nDimension == 3 || input->nDimension == 4) )
		THError("3D or 4D (batch mode) tensor expected");

	if (input->nDimension == 4)
	{
		nbatch = input->size[0];
		dimw++;
		dimh++;
		dimc++;
	}

	channels = input->size[dimc];
	inputWidth = input->size[dimw];
	inputHeight = input->size[dimh];

	if (input->nDimension == 3)
		THFloatTensor_resize3d(output, channels, inputHeight, inputWidth);
	else
		THFloatTensor_resize4d(output, input->size[0], channels, inputHeight, inputWidth);

	THFloatTensor *scale = THFloatTensor_new();
	THFloatTensor_resizeAs(scale, input);
	THFloatTensor_zero(scale);

	input_data = THFloatTensor_data(input);
	output_data = THFloatTensor_data(output);
	scale_data = THFloatTensor_data(scale);

	/* calculate square of input in output */

	long l;
//#pragma omp parallel for private(l)
	for (l = 0; l < nbatch * channels * inputWidth * inputHeight; l++)
		output_data[l] = input_data[l] * input_data[l];

	int prePad = ((size - 1) / 2) + 1;
	int prePadCrop = prePad > channels ? channels : prePad;

	int ci, wi, hi, bi;
	long cs, ws, hs, bs;

	cs = input->stride[dimc];
	ws = input->stride[dimw];
	hs = input->stride[dimh];
	bs = input->nDimension == 3 ? 0 : input->stride[0];

	/* first feature map normalization */
	for (ci = 0; ci < prePadCrop; ci++) {
//#pragma omp parallel for private(wi)
		for (wi = 0; wi < inputWidth; wi++) {
			for (hi = 0; hi < inputHeight; hi++) {
				for (bi = 0; bi < nbatch; bi++) {
					scale_data[ci * cs + wi * ws + hi * hs + bi * bs] +=
						output_data[ci * cs + wi * ws + hi * hs + bi * bs];
				}
			}
		}
	}

	/* reuse normalizations for further channels */
	for (ci = 1; ci < channels; ci++) {
//#pragma omp parallel for private(wi)
		for (wi = 0; wi < inputWidth; wi++) {
			for (hi = 0; hi < inputHeight; hi++) {
				for (bi = 0; bi < nbatch; bi++) {
					scale_data[ci * cs + wi * ws + hi * hs + bi * bs] =
						scale_data[(ci - 1) * cs + wi * ws + hi * hs + bi * bs];
					if (ci + prePad - 1 < channels) {
						scale_data[ci * cs + wi * ws + hi * hs + bi * bs] +=
							input_data[(ci + prePad - 1) * cs + wi * ws + hi * hs + bi * bs];
					}
					if (ci - prePad >= 0) {
						scale_data[ci * cs + wi * ws + hi * hs + bi * bs] -=
							input_data[(ci - prePad) * cs + wi * ws + hi * hs + bi * bs];
					}
				}
			}
		}	
	}

//#pragma omp parallel for private(l)
	for (l = 0; l < nbatch * channels * inputWidth * inputHeight; l++) {
		scale_data[l] *= (alpha/size);
		scale_data[l] += k;
		output_data[l] = pow(scale_data[l], -beta);
		output_data[l] *= input_data[l];
	}

	THFloatTensor_free(scale);
	return output;
}
