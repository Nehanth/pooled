// q38dump: run fixed token sequences through llama.cpp and write named graph tensors as raw .f32 files, plus the
// PLE n-gram row indices (the I32 input of the get_rows on per_layer_token_embd). It is eval-callback with files
// instead of printed sums, for tests/reference/ref_q38flash.mjs and tests/fixtures/q38flash/ngram_rows.json.
//
// Build: copy this directory into a llama.cpp checkout that has qwen4exp (master >= 6c84c7d) as examples/q38dump,
// add "add_subdirectory(q38dump)" next to eval-callback in examples/CMakeLists.txt, then
//   cmake -B build -DGGML_CUDA=ON && cmake --build build --target q38dump
// Run on the CPU backend, f32 KV, no flash attention and no repacking, which is what the reference emulates.
// The dumps the reference was checked against were made with exactly this, from ~/llm/models/q38flash, and live
// in ~/llm/models/q38flash/dumps (689 MB, 4 sequences; re-running it reproduces them byte for byte):
//   cd ~/llm/models/q38flash
//   export Q38_OUT=dumps
//   export Q38_TOKENS="760,6511,314,9338,369,11751;248044,760,6511,248044,248044,314,9338,248044,369;760,6511,314|9338,248044,369,11751|11751|248044|5|6;248319,248056,0,1,248045,248046,248044,7"
//   export Q38_FILTER='^(model\.input_embed|hc_init|ple_|hc_mixed|hc_inject|hc_combine|hc_gate|l_last|attn_output|attn_gated|linear_attn_out|final_output|z-|ffn_moe_out|ffn_out|ffn_moe_topk|ffn_moe_weights_norm|ffn_shexp_gated|result_norm|result_output)'
//   q38dump -m Qwen3.8-Flash-Next-Q4_0-00001-of-00003.gguf -dev none -ngl 0 -fa off -ctk f32 -ctv f32 -nr -c 256 -t 16 --fit off
// ';' separates sequences (memory is cleared between them), '|' splits one sequence into consecutive decode
// calls, so the later chunks read their n-gram predecessors and PLE conv history from the cache.
// Files: <out>/s<seq>/c<chunk>/<name>.<occurrence>.f32 (or .i32) and shapes.tsv per chunk. A name repeats when
// llama.cpp names two tensors alike (each layer has two hc_mixed); <occurrence> counts them in graph order.
// Each chunk also gets its tokens in tokens.txt, which the reference reads.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

struct dump_state {
    std::string dir;
    std::regex  filter;
    std::map<std::string, int> seen;
    FILE * shapes = nullptr;
    std::vector<uint8_t> buf;
};

static void write_tensor(dump_state & st, const ggml_tensor * t, const std::string & name, bool as_i32) {
    const size_t nbytes = ggml_nbytes(t);
    st.buf.resize(nbytes);
    ggml_backend_tensor_get(t, st.buf.data(), 0, nbytes);
    const int64_t n = ggml_nelements(t);
    std::vector<float>   f(as_i32 ? 0 : n);
    std::vector<int32_t> iv(as_i32 ? n : 0);
    int64_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
    for (int64_t i0 = 0; i0 < t->ne[0]; i0++, k++) {
        const uint8_t * p = st.buf.data() + i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0*t->nb[0];
        if (as_i32) { iv[k] = *(const int32_t *) p; continue; }
        switch (t->type) {
            case GGML_TYPE_F32:  f[k] = *(const float *) p; break;
            case GGML_TYPE_F16:  f[k] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p); break;
            case GGML_TYPE_BF16: f[k] = ggml_bf16_to_fp32(*(const ggml_bf16_t *) p); break;
            default: return; // quantized views are weights, not activations
        }
    }
    const int occ = st.seen[name]++;
    const std::string file = st.dir + "/" + name + "." + std::to_string(occ) + (as_i32 ? ".i32" : ".f32");
    FILE * fp = fopen(file.c_str(), "wb");
    if (as_i32) fwrite(iv.data(), 4, n, fp); else fwrite(f.data(), 4, n, fp);
    fclose(fp);
    fprintf(st.shapes, "%s.%d\t%s\t%lld\t%lld\t%lld\t%lld\t%s\n", name.c_str(), occ, ggml_op_desc(t),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_type_name(t->type));
}

static bool is_ple_gather(const ggml_tensor * t) {
    return t->op == GGML_OP_GET_ROWS && t->src[0] && strcmp(t->src[0]->name, "per_layer_token_embd.weight") == 0;
}

static bool cb_eval(ggml_tensor * t, bool ask, void * ud) {
    auto & st = *(dump_state *) ud;
    const bool want = std::regex_search(t->name, st.filter) || is_ple_gather(t);
    if (ask) return want;
    if (!want) return true;
    if (is_ple_gather(t)) write_tensor(st, t->src[1], "ple_rows", true);
    if (std::regex_search(t->name, st.filter)) write_tensor(st, t, t->name, t->type == GGML_TYPE_I32);
    return true;
}

static std::vector<std::vector<std::vector<llama_token>>> parse_tokens(const std::string & s) {
    std::vector<std::vector<std::vector<llama_token>>> seqs;
    std::stringstream ss(s);
    std::string seq;
    while (std::getline(ss, seq, ';')) {
        std::vector<std::vector<llama_token>> chunks;
        std::stringstream cs(seq);
        std::string chunk;
        while (std::getline(cs, chunk, '|')) {
            std::vector<llama_token> ids;
            std::stringstream is(chunk);
            std::string id;
            while (std::getline(is, id, ',')) if (!id.empty()) ids.push_back(atoi(id.c_str()));
            chunks.push_back(ids);
        }
        seqs.push_back(chunks);
    }
    return seqs;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    const char * out = getenv("Q38_OUT");
    const char * tok = getenv("Q38_TOKENS");
    const char * flt = getenv("Q38_FILTER");
    if (!out || !tok) { fprintf(stderr, "set Q38_OUT and Q38_TOKENS\n"); return 1; }

    dump_state st;
    st.filter = std::regex(flt ? flt : "^$");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto * ctx = init->context();
    if (!ctx) return 1;

    const auto seqs = parse_tokens(tok);
    mkdir(out, 0755);
    for (size_t s = 0; s < seqs.size(); s++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos pos = 0;
        for (size_t c = 0; c < seqs[s].size(); c++) {
            st.dir = std::string(out) + "/s" + std::to_string(s);
            mkdir(st.dir.c_str(), 0755);
            st.dir += "/c" + std::to_string(c);
            mkdir(st.dir.c_str(), 0755);
            st.seen.clear();
            st.shapes = fopen((st.dir + "/shapes.tsv").c_str(), "w");
            FILE * tf = fopen((st.dir + "/tokens.txt").c_str(), "w");
            for (size_t i = 0; i < seqs[s][c].size(); i++) fprintf(tf, i ? ",%d" : "%d", seqs[s][c][i]);
            fprintf(tf, "\n");
            fclose(tf);
            common_batch batch(ctx);
            for (llama_token id : seqs[s][c]) batch.add(id, pos++, 0, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { fprintf(stderr, "decode failed\n"); return 1; }
            fclose(st.shapes);
            fprintf(stderr, "seq %zu chunk %zu: %zu tokens\n", s, c, seqs[s][c].size());
        }
    }
    llama_backend_free();
    return 0;
}
