#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	assert(argc >= 4);
	int i, j, model_index = argc > 4? atoi(argv[4]) : 0;
    assert(model_index >= 0);
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* seq = ccv_dpm_sparse_detect_objects(image, argv[2], argv[3], model_index, ccv_dpm_default_params);
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
			size_t len = 1024;
			char* file = (char*)malloc(len);
			ssize_t read;
			while((read = getline(&file, &len, r)) != -1)
			{
				while(read > 1 && isspace(file[read - 1]))
					read--;
				file[read] = 0;
				image = 0;
				ccv_read(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				assert(image != 0);
				ccv_array_t* seq = ccv_dpm_sparse_detect_objects(image, argv[2], argv[3], model_index, ccv_dpm_default_params);
				if (seq != 0)
				{
					for (i = 0; i < seq->rnum; i++)
					{
						ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
						printf("%s %d %d %d %d %f %d\n", file, comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
						for (j = 0; j < comp->pnum; j++)
							printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
					}
					ccv_array_free(seq);
				}
				ccv_matrix_free(image);
			}
			free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	return 0;
}