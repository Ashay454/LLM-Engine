#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
} TransformerWeights;

typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
} Tokenizer;

typedef struct {
    float prob;
    int index;
} TokenProb;

int compare_tokens(const void* a, const void* b) {
    TokenProb* pa = (TokenProb*)a;
    TokenProb* pb = (TokenProb*)b;
    if (pa->prob < pb->prob) return 1;
    if (pa->prob > pb->prob) return -1;
    return 0;
}

void setup_console() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    GetConsoleMode(hOut, &dwMode);
    dwMode |= 0x0004;
    SetConsoleMode(hOut, dwMode);
#endif
}

void print_dashboard(TokenProb* probs, Tokenizer* t, int n_show, int current_pos) {
    printf("\033[H");
    printf("\033[1;36m==================================================\033[0m\033[K\n");
    printf(" \033[1;37mENGINE KERNEL:\033[0m \033[1;32mONLINE\033[0m   |   \033[1;37mTOKEN POS:\033[0m %d/50\033[K\n", current_pos);
    printf("\033[1;36m==================================================\033[0m\033[K\n");
    printf("\n\033[1;33m[ REAL-TIME PROBABILITY ]\033[0m\033[K\n");

    for (int i = 0; i < n_show; i++) {
        char* token_str = t->vocab[probs[i].index];
        float p = probs[i].prob * 100.0f;
        
        printf(" %-12s \033[1;37m|\033[0m", token_str);
        
        int bar_len = (int)(p / 100.0f * 25);
        
        if (i == 0) printf("\033[1;32m");
        else if (i == 1) printf("\033[1;33m");
        else printf("\033[1;34m");
        
        for (int j = 0; j < bar_len; j++) printf("#");
        printf("\033[0m");
        printf(" %.2f%%\033[K\n", p);
    }
    printf("\n\033[1;36m--------------------------------------------------\033[0m\033[K\n");
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = x[j] * weight[j] * ss;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void forward(TransformerWeights* w, RunState* s, Config* p, int token, int pos) {
    float* content_row = w->token_embedding_table + token * p->dim;
    memcpy(s->x, content_row, p->dim * sizeof(float));

    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int head_size = dim / p->n_heads;

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, w->rms_att_weight + l * dim, dim);

        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float q0 = s->q[i];
            float q1 = s->q[i + 1];
            
            s->q[i] = q0 * fcr - q1 * fci;
            s->q[i + 1] = q0 * fci + q1 * fcr;
            
            if (i < kv_dim) {
                float k0 = s->k[i];
                float k1 = s->k[i + 1];
                s->k[i] = k0 * fcr - k1 * fci;
                s->k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            float* xb2 = s->xb2 + h * head_size;
            
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);
            memset(xb2, 0, head_size * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb2[i] += a * v[i];
                }
            }
        }

        matmul(s->xb, s->xb2, w->wo + l * dim * dim, dim, dim);
        
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + l * dim * p->hidden_dim, dim, p->hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * p->hidden_dim, dim, p->hidden_dim);

        for (int i = 0; i < p->hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            s->hb[i] = val * s->hb2[i];
        }

        matmul(s->xb, s->hb, w->w2 + l * p->hidden_dim * dim, p->hidden_dim, dim);
        
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    matmul(s->logits, s->x, w->token_embedding_table, dim, p->vocab_size);
}

void malloc_run_state(RunState* s, Config* p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    s->x = calloc(dim, sizeof(float));
    s->xb = calloc(dim, sizeof(float));
    s->xb2 = calloc(dim, sizeof(float));
    s->hb = calloc(hidden_dim, sizeof(float));
    s->hb2 = calloc(hidden_dim, sizeof(float));
    s->q = calloc(dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
}

void load_model(Config* c, TransformerWeights* w, float** d, char* p) {
    FILE *f = fopen(p, "rb");
    if (!f) exit(1);
    
    fread(c, sizeof(Config), 1, f);
    fseek(f, 0, SEEK_END);
    long s = ftell(f);
    fseek(f, sizeof(Config), SEEK_SET);
    
    *d = malloc(s - sizeof(Config));
    fread(*d, 1, s - sizeof(Config), f);
    fclose(f);

    float* ptr = *d;
    w->token_embedding_table = ptr; ptr += c->vocab_size * c->dim;
    w->rms_att_weight = ptr; ptr += c->n_layers * c->dim;
    w->wq = ptr; ptr += c->n_layers * c->dim * c->dim;
    w->wk = ptr; ptr += c->n_layers * c->dim * c->dim;
    w->wv = ptr; ptr += c->n_layers * c->dim * c->dim;
    w->wo = ptr; ptr += c->n_layers * c->dim * c->dim;
    w->rms_ffn_weight = ptr; ptr += c->n_layers * c->dim;
    w->w1 = ptr; ptr += c->n_layers * c->dim * c->hidden_dim;
    w->w2 = ptr; ptr += c->n_layers * c->hidden_dim * c->dim;
    w->w3 = ptr; ptr += c->n_layers * c->dim * c->hidden_dim;
    w->rms_final_weight = ptr;
}

void build_tokenizer(Tokenizer* t, char* p, int s) {
    FILE *f = fopen(p, "rb");
    if (!f) exit(1);
    
    t->vocab_size = s;
    t->vocab = malloc(s * sizeof(char*));
    t->vocab_scores = malloc(s * sizeof(float));
    
    int len;
    fread(&len, sizeof(int), 1, f);
    
    for (int i = 0; i < s; i++) {
        fread(&t->vocab_scores[i], sizeof(float), 1, f);
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = malloc(len + 1);
        fread(t->vocab[i], len, 1, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

int main() {
    setup_console();

    Config config;
    TransformerWeights weights;
    float* data_memory;
    RunState state;
    Tokenizer tokenizer;

    load_model(&config, &weights, &data_memory, "stories15M.bin");
    build_tokenizer(&tokenizer, "tokenizer.bin", config.vocab_size);
    malloc_run_state(&state, &config);

    printf("\033[2J");

    int token = 1;
    char story_buffer[1024] = "";
    TokenProb* probs = malloc(config.vocab_size * sizeof(TokenProb));

    for (int pos = 0; pos < 50; pos++) {
        forward(&weights, &state, &config, token, pos);

        softmax(state.logits, config.vocab_size);
        
        for (int i = 0; i < config.vocab_size; i++) {
            probs[i].index = i;
            probs[i].prob = state.logits[i];
        }
        
        qsort(probs, config.vocab_size, sizeof(TokenProb), compare_tokens);

        print_dashboard(probs, &tokenizer, 5, pos);

        int next_token = probs[0].index;
        char* word = tokenizer.vocab[next_token];
        strcat(story_buffer, word);

        printf("\033[1;37mSTORY STREAM:\033[0m\033[K\n");
        printf("\033[1;32m%s\033[0m\033[5m_\033[0m\033[K", story_buffer);
        fflush(stdout);

        token = next_token;

#ifdef _WIN32
        Sleep(200);
#else
        usleep(200000);
#endif
    }

    printf("\n\n");
    free(data_memory);
    free(state.key_cache);
    free(state.value_cache);
    free(probs);
    
    return 0;
}