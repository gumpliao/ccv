#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void read_sparse_dict(const char* directory, ccv_dpm_root_classifier_t* sparse_classifier)
{
    FILE* r = fopen(directory, "r");
    if (r == 0)
        return;
    int rows, cols;
    fscanf(r, "%d %d %d", &rows, &cols, &sparse_classifier->count);
    
    ccv_dpm_part_classifier_t* part_classifier = (ccv_dpm_part_classifier_t*)ccmalloc(sizeof(ccv_dpm_part_classifier_t) * sparse_classifier->count);
    int j, k;
    for (j = 0; j < sparse_classifier->count; j++)
    {
        part_classifier[j].w = ccv_dense_matrix_new(rows, cols, CCV_32F | 31, 0, 0);
        for (k = 0; k < rows * cols * 31; k++)
            fscanf(r, "%f", &part_classifier[j].w->data.f32[k]);
        ccv_make_matrix_immutable(part_classifier[j].w);
    }
    sparse_classifier->part = part_classifier;
    fclose(r);
}

int main(int argc, char** argv)
{
	assert(argc >= 4);
	int i, j, model_index = argc > 4? atoi(argv[4]) : 0, num_models, num_filters;
    assert(model_index >= 0);
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE);
    
    /* read dictionary of sparse filters and store the data into sparse_classifier */
    ccv_dpm_root_classifier_t* sparse_classifier = (ccv_dpm_root_classifier_t*)alloca(sizeof(ccv_dpm_root_classifier_t));
    memset(sparse_classifier, 0, sizeof(ccv_dpm_root_classifier_t));
    read_sparse_dict(argv[2], sparse_classifier);
    
    /* reading from alpha vectors file:
     * number of models involved (first int in the file)
     * paths to each model file */
    FILE* r = fopen(argv[3], "r");
    if (r == 0)
        return 0;
    fscanf(r, "%d", &num_models);
    ccv_dpm_mixture_model_t** models = (ccv_dpm_mixture_model_t**)alloca(sizeof(ccv_dpm_mixture_model_t*) * num_models);
    size_t len = 1024;
    char* line = (char*)malloc(len);
    ssize_t read;
    for (i = 0; i < num_models; i++)
    {
        if ((read = getline(&line, &len, r)) != -1)
        {
            if (line[0] != '\n')
            {
                while(read > 1 && isspace(line[read - 1]))
                    read--;
                line[read] = 0;
                models[i] = ccv_dpm_read_mixture_model(line);
            }
            else
                i--;
        }
    }
    
    /* reading number of part filters (num_filters) and dictionary size (j) from alpha file */
    fscanf(r, "%d %d", &num_filters, &j);
    ccv_dense_matrix_t* alpha = ccv_dense_matrix_new(num_filters, j, CCV_32F | CCV_C1, 0, 0);
    /* storing alpha matrix data */
    for (i = 0; i < num_filters*j; i++)
        fscanf(r, "%f", &alpha->data.f32[i]);
    fclose(r);
    
    /* check consistency of size between alpha and sparse dicts */
    assert(alpha->cols == sparse_classifier->count);
    
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* seq = ccv_dpm_sparse_detect_objects(image, sparse_classifier, alpha, models, num_models, model_index, ccv_dpm_default_params);
		elapsed_time = get_current_time() - elapsed_time;
		if (seq)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
				printf("%d %d %d %d %f %d\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
				for (j = 0; j < comp->pnum; j++)
					printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
			}
			printf("total : %d in time %dms\n", seq->rnum, elapsed_time);
			ccv_array_free(seq);
		} else {
			printf("elapsed time %dms\n", elapsed_time);
		}
		ccv_matrix_free(image);
	} else {
		FILE* r = fopen(argv[1], "rt");
		if (argc == 6)
			chdir(argv[5]);
		if(r)
		{
			while((read = getline(&line, &len, r)) != -1)
			{
				while(read > 1 && isspace(line[read - 1]))
					read--;
				line[read] = 0;
				image = 0;
				ccv_read(line, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				assert(image != 0);
				ccv_array_t* seq = ccv_dpm_sparse_detect_objects(image, sparse_classifier, alpha, models, num_models, model_index, ccv_dpm_default_params);
				if (seq != 0)
				{
					for (i = 0; i < seq->rnum; i++)
					{
						ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
						printf("%s %d %d %d %d %f %d\n", line, comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
						for (j = 0; j < comp->pnum; j++)
							printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
					}
					ccv_array_free(seq);
				}
				ccv_matrix_free(image);
			}
			fclose(r);
		}
	}
    free(line);
	ccv_drain_cache();
    for (i = 0; i < num_models; i++)
        ccv_dpm_mixture_model_free(models[i]);
    ccv_matrix_free(alpha);
    ccv_dpm_part_classifier_t* part_classifier = sparse_classifier->part;
    for (j = 0; j < sparse_classifier->count; j++)
        ccv_matrix_free(part_classifier[j].w);
    ccfree(part_classifier);
	return 0;
}