#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <jpeglib.h>
#include <time.h>

#define N 64       
#define MAX_IMAGES 200   
#define TRAIN_SPLIT 0.8      
#define LEARNING_RATE 0.01  
#define EPOCHS 100  
#define TYPE 2 // 1 GD, 2 SGD, 3 ADAM

typedef struct {
    double *data;
    int label; 
} Image;

typedef struct {
    Image *images;
    int count;
} Dataset;

Image read_image(const char *filepath) {
    Image img = {NULL, 0};
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Dosya açılamadı: %s\n", filepath);
        return img;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;

    if (channels != 3) { // RGB formatı kontrolü
        fprintf(stderr, "Beklenmeyen kanal sayısı: %d\n", channels);
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        return img;
    }

    // Tüm resim verisini belleğe yükle
    unsigned char *raw_data = (unsigned char *)malloc(width * height * channels);
    unsigned char *row_pointer[1];
    for (int i = 0; i < height; i++) {
        row_pointer[0] = raw_data + i * width * channels;
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    // Normalize edilmiş gri tonlamalı resim için bellek ayır
    img.data = (double *)malloc(N * N * sizeof(double));
    double *resized = img.data;

    // Ölçekleme faktörleri
    double scale_x = (double)width / N;
    double scale_y = (double)height / N;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int src_x = (int)(x * scale_x);
            int src_y = (int)(y * scale_y);
            int src_index = (src_y * width + src_x) * channels;

            // R, G ve B kanallarının ortalaması ile gri tonlama
            int gray = (raw_data[src_index] + raw_data[src_index + 1] + raw_data[src_index + 2]) / 3;
            resized[y * N + x] = gray / 255.0;
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
    free(raw_data);

    return img;
}

// Veri kümesi yükleme
Dataset load_dataset(const char *a_dir, const char *b_dir, int max_samples) {
    Dataset dataset;
    dataset.images = (Image *)malloc(2 * max_samples * sizeof(Image));
    dataset.count = 0;

    char filepath[256];
    for (int i = 0; i < max_samples; i++) {
        sprintf(filepath, "%s/damaged_%d.jpg", a_dir, i + 1);
        Image img_a = read_image(filepath);
        if (img_a.data) {
            img_a.label = 1;
            dataset.images[dataset.count++] = img_a;
        }

        sprintf(filepath, "%s/dog.%d.jpg", b_dir, i + 1);
        Image img_b = read_image(filepath);
        if (img_b.data) {
            img_b.label = -1;
            dataset.images[dataset.count++] = img_b;
        }
    }

    return dataset;
}

// Eğitim ve test verisine ayırma
void split_dataset(Dataset *dataset, Dataset *train, Dataset *test) {
    int train_size = dataset->count * TRAIN_SPLIT;
    train->images = dataset->images;
    train->count = train_size;
    test->images = dataset->images + train_size;
    test->count = dataset->count - train_size;
}

// Modeli eğit ve değerlendir


void gd_train_and_evaluate(Dataset *train, Dataset *test, double *weights, const char *filename) {
    int input_size = N * N + 1; // Bias dahil

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Dosya açılamadı: %s\n", filename);
        return;
    }
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double train_loss = 0, test_loss = 0;
        int correct = 0;

        // Zaman ölçümü başlangıcı
        clock_t start_time = clock();


        // Ağırlıkları dosyaya kaydet
        for (int i = 0; i < input_size; i++) {
            fprintf(file, "%f ", weights[i]);
        }
        fprintf(file, "\n");

        // Eğitim aşaması
        for (int i = 0; i < train->count; i++) {
            Image *img = &train->images[i];
            double *x = img->data;

            // wx hesapla
            double wx = weights[0]; // Bias
            for (int j = 0; j < N * N; j++) {
                wx += weights[j + 1] * x[j];
            }

            // Çıkış ve hata
            double y_pred = tanh(wx);
            double error = img->label - y_pred;
            train_loss += error * error;

            // Ağırlıkları güncelle
            weights[0] += LEARNING_RATE * error; // Bias
            for (int j = 0; j < N * N; j++) {
                weights[j + 1] += LEARNING_RATE * error * x[j];
            }
        }

        train_loss /= train->count;

        // Test aşaması
        for (int i = 0; i < test->count; i++) {
            Image *img = &test->images[i];
            double *x = img->data;

            // wx hesapla
            double wx = weights[0];
            for (int j = 0; j < N * N; j++) {
                wx += weights[j + 1] * x[j];
            }

            double y_pred = tanh(wx);
            double error = img->label - y_pred;
            test_loss += error * error;

            // Tahmin doğruluğunu kontrol et
            if ((y_pred > 0 && img->label == 1) || (y_pred < 0 && img->label == -1)) {
                correct++;
            }
        }

        test_loss /= test->count;
        double accuracy = (double)correct / test->count * 100.0;


        // Zaman ölçümü bitişi
        clock_t end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Epoch %d: Train Loss = %.4f, Test Loss = %.4f, Accuracy = %.2f%%, Time = %.2fs\n",
               epoch + 1, train_loss, test_loss, accuracy, epoch_time);

        // Zaman ve loss'u dosyaya kaydet
        fprintf(file, "Epoch %d, Train Loss: %.4f, Test Loss: %.4f, Time: %.4fs\n", 
                epoch + 1, train_loss, test_loss, epoch_time);
    }
    fclose(file);
}

void sgd_train_and_evaluate(Dataset *train, Dataset *test, double *weights, const char *filename, int batch_size) {
    int input_size = N * N + 1; // Bias dahil
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Dosya açılamadı: %s\n", filename);
        return;
    }

    int num_batches = (train->count + batch_size - 1) / batch_size; // Minibatch sayısı

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double train_loss = 0, test_loss = 0;
        int correct = 0;

        // Zaman ölçümü başlangıcı
        clock_t start_time = clock();

        // Ağırlıkları dosyaya kaydet
        for (int i = 0; i < input_size; i++) {
            fprintf(file, "%f ", weights[i]);
        }
        fprintf(file, "\n");

        // Eğitim aşaması
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * batch_size;
            int end_idx = (start_idx + batch_size > train->count) ? train->count : start_idx + batch_size;

            for (int i = start_idx; i < end_idx; i++) {
                Image *img = &train->images[i];
                double *x = img->data;

                double wx = weights[0];
                for (int j = 0; j < N * N; j++) {
                    wx += weights[j + 1] * x[j];
                }

                double y_pred = tanh(wx);
                double error = img->label - y_pred;
                train_loss += error * error;

                weights[0] += LEARNING_RATE * error; // Bias güncelleme
                for (int j = 0; j < N * N; j++) {
                    weights[j + 1] += LEARNING_RATE * error * x[j];
                }
            }
        }

        train_loss /= train->count;

        // Test aşaması
        for (int i = 0; i < test->count; i++) {
            Image *img = &test->images[i];
            double *x = img->data;

            double wx = weights[0];
            for (int j = 0; j < N * N; j++) {
                wx += weights[j + 1] * x[j];
            }

            double y_pred = tanh(wx);
            double error = img->label - y_pred;
            test_loss += error * error;

            if ((y_pred > 0 && img->label == 1) || (y_pred < 0 && img->label == -1)) {
                correct++;
            }
        }

        test_loss /= test->count;
        double accuracy = (double)correct / test->count * 100.0;

        // Zaman ölçümü bitişi
        clock_t end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Epoch %d: Train Loss = %.4f, Test Loss = %.4f, Accuracy = %.2f%%, Time = %.2fs\n",
               epoch + 1, train_loss, test_loss, accuracy, epoch_time);

        // Zaman ve loss'u dosyaya kaydet
        fprintf(file, "Epoch %d, Train Loss: %.4f, Test Loss: %.4f, Time: %.4fs\n", 
                epoch + 1, train_loss, test_loss, epoch_time);
    }

    fclose(file);
}

void adam_train_and_evaluate(Dataset *train, Dataset *test, double *weights, const char *filename) {
    int input_size = N * N + 1; // Bias dahil

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Dosya açılamadı: %s\n", filename);
        return;
    }

    double beta2 = 0.999;        // İkinci moment için beta değeri
    double epsilon = 1e-8;      // Küçük bir sayı, sıfır bölme hatalarını engeller
    double *v = (double *)calloc(input_size, sizeof(double));  // İkinci moment (gradyan karelerinin ortalaması)

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double train_loss = 0, test_loss = 0;
        int correct = 0;

        // Zaman ölçümü başlangıcı
        clock_t start_time = clock();


        // Ağırlıkları dosyaya kaydet
        for (int i = 0; i < input_size; i++) {
            fprintf(file, "%f ", weights[i]);
        }
        fprintf(file, "\n");

        // Eğitim aşaması
        for (int i = 0; i < train->count; i++) {
            Image *img = &train->images[i];
            double *x = img->data;

            // wx hesapla
            double wx = weights[0]; // Bias
            for (int j = 0; j < N * N; j++) {
                wx += weights[j + 1] * x[j];
            }

            // Çıkış ve hata
            double y_pred = tanh(wx);
            double error = img->label - y_pred;
            train_loss += error * error;

            // Gradyan hesaplama ve güncelleme
            for (int j = 0; j < input_size; j++) {
                double grad = (j == 0) ? error : error * x[j - 1];  // Bias için error, diğerleri için x_j * error

                // İkinci moment güncellemesi (v)
                v[j] = beta2 * v[j] + (1 - beta2) * grad * grad;

                // Bias düzeltmesi (v_t için)
                double v_hat = v[j] / (1 - pow(beta2, epoch + 1));  // Bias düzeltmesi

                // Ağırlık güncellemesi
                weights[j] += LEARNING_RATE * grad / (sqrt(v_hat) + epsilon);
            }
        }

        train_loss /= train->count;

        // Test aşaması
        for (int i = 0; i < test->count; i++) {
            Image *img = &test->images[i];
            double *x = img->data;

            // wx hesapla
            double wx = weights[0];
            for (int j = 0; j < N * N; j++) {
                wx += weights[j + 1] * x[j];
            }

            double y_pred = tanh(wx);
            double error = img->label - y_pred;
            test_loss += error * error;

            // Tahmin doğruluğunu kontrol et
            if ((y_pred > 0 && img->label == 1) || (y_pred < 0 && img->label == -1)) {
                correct++;
            }
        }

        test_loss /= test->count;
        double accuracy = (double)correct / test->count * 100.0;

        // Zaman ölçümü bitişi
        clock_t end_time = clock();
        double epoch_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Epoch %d: Train Loss = %.4f, Test Loss = %.4f, Accuracy = %.2f%%, Time = %.2fs\n",
               epoch + 1, train_loss, test_loss, accuracy, epoch_time);

        // Zaman ve loss'u dosyaya kaydet
        fprintf(file, "Epoch %d, Train Loss: %.4f, Test Loss: %.4f, Time: %.4fs\n", 
                epoch + 1, train_loss, test_loss, epoch_time);
    }

    fclose(file);
    free(v); // Belleği temizle
}


// Ana fonksiyon
int main() {
    const char *a_dir = "./images/eggs";
    const char *b_dir = "./images/dogs";

    // Veri kümesini yükle
    Dataset dataset = load_dataset(a_dir, b_dir, MAX_IMAGES);

    // Eğitim ve test kümelerine ayır
    Dataset train, test;
    split_dataset(&dataset, &train, &test);

    // Ağırlıkları başlat
    int input_size = N * N + 1; // Bias dahil
    double *weights = (double *)calloc(input_size, sizeof(double));

    int batch_size = 16; // Örneğin 16

    // Modeli eğit ve değerlendir

    // 5 farklı başlangıç için modeli eğit ve ağırlıkları kaydet
        for (int i = 0; i < 5; i++) {
            // Ağırlıkları rastgele başlat
            for (int j = 0; j < input_size; j++) {
                weights[j] = (rand() / (double)RAND_MAX) * 0.1; // Küçük rastgele değerler
            }
            // Her bir başlangıç için farklı bir dosyaya yaz
            char filename[256];

            if (TYPE== 1) {
                sprintf(filename, "weights_run_%d.txt", i + 1);
                gd_train_and_evaluate(&train, &test, weights, filename);
            } if (TYPE == 2) {
                sprintf(filename, "weights_run_%d.txt", i + 6);
                sgd_train_and_evaluate(&train, &test, weights, filename, batch_size);
            } if (TYPE == 3) {
                sprintf(filename, "weights_run_%d.txt", i + 11);
                adam_train_and_evaluate(&train, &test, weights, filename);
            }

            
        }


    // Belleği temizle
    for (int i = 0; i < dataset.count; i++) {
        free(dataset.images[i].data);
    }
    free(dataset.images);
    free(weights);

    return 0;
}
