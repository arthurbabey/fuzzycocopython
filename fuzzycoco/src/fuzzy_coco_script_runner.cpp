

#include <iostream>
#include <duktape.h>
#include "fuzzy_coco_script_runner.h"

static FuzzyCocoScriptRunner* REGISTERED_RUNNER = 0;

// C function wrapper to be called from JS
static duk_ret_t _duktape_setParams(duk_context * ctx)
{
    if (REGISTERED_RUNNER == 0) return DUK_RET_ERROR;
    ScriptParams sp;
    FuzzyCocoParams& p = sp.coco;
    int i = 0;
    // sp.coco.experimentName = duk_safe_to_string(ctx, i++);
    // sp.coco.fixedVars = duk_to_boolean(ctx, i++);

    // FuzzySystemParams& f = sp.coco.fs_params;
    sp.coco.global_params.nb_rules = duk_to_int(ctx, i++);
    sp.coco.global_params.nb_max_var_per_rule = duk_to_int(ctx, i++);
    sp.nbOutputVars = duk_to_int(ctx, i++);
    sp.coco.input_vars_params.nb_sets = duk_to_int(ctx, i++);
    sp.coco.output_vars_params.nb_sets = duk_to_int(ctx, i++);
    sp.coco.input_vars_params.nb_bits_vars = duk_to_int(ctx, i++);
    sp.coco.output_vars_params.nb_bits_vars = duk_to_int(ctx, i++);
    sp.coco.input_vars_params.nb_bits_sets = duk_to_int(ctx, i++);
    sp.coco.output_vars_params.nb_bits_sets = duk_to_int(ctx, i++);
    sp.coco.input_vars_params.nb_bits_pos = duk_to_int(ctx, i++);
    sp.coco.output_vars_params.nb_bits_pos = duk_to_int(ctx, i++);

    sp.coco.global_params.max_generations = duk_to_int(ctx, i++);
    sp.coco.global_params.max_fitness = duk_to_number(ctx, i++);

    EvolutionParams& p1 = sp.coco.rules_params;
    // p1.maxGen = duk_to_int(ctx, i++);
    // p1.maxFit = duk_to_number(ctx, i++);
    p1.elite_size = duk_to_int(ctx, i++);
    p1.pop_size = duk_to_int(ctx, i++);
    p1.cx_prob = duk_to_number(ctx, i++);
    p1.mut_flip_genome = duk_to_number(ctx, i++);
    p1.mut_flip_bit = duk_to_number(ctx, i++);

    EvolutionParams& p2 = sp.coco.mfs_params;
    // p2.maxGen = duk_to_int(ctx, i++);
    // p2.maxFit = duk_to_number(ctx, i++);
    p2.elite_size = duk_to_int(ctx, i++);
    p2.pop_size = duk_to_int(ctx, i++);
    p2.cx_prob = duk_to_number(ctx, i++);
    p2.mut_flip_genome = duk_to_number(ctx, i++);
    p2.mut_flip_bit = duk_to_number(ctx, i++);

    FuzzySystemMetrics& w = sp.coco.metrics_weights;
    w.sensitivity = duk_to_number(ctx, i++);
    w.specificity = duk_to_number(ctx, i++);
    w.accuracy = duk_to_number(ctx, i++);
    w.ppv = duk_to_number(ctx, i++);
    w.rmse = duk_to_number(ctx, i++);
    w.rrse = duk_to_number(ctx, i++);
    w.rae = duk_to_number(ctx, i++);
    w.mse = duk_to_number(ctx, i++);
    w.distanceThreshold = duk_to_number(ctx, i++);
    w.distanceMinThreshold = duk_to_number(ctx, i++);
    w.nb_vars = duk_to_number(ctx, i++);
    w.overLearn = duk_to_number(ctx, i++);

    double defuzz_threshold = duk_to_number(ctx, i++);
    sp.coco.output_vars_defuzz_thresholds.push_back(defuzz_threshold);
    sp.coco.defuzz_threshold_activated = duk_to_boolean(ctx, i++);

    REGISTERED_RUNNER->setParams(sp);

    return 0;
}

// static duk_ret_t _duktape_print(duk_context * ctx)
// {
//     cerr << duk_safe_to_string(ctx, -1);
//     return 0;
// }

static duk_ret_t native_print(duk_context *ctx) {
    duk_push_string(ctx, " ");
    duk_insert(ctx, 0);
    duk_join(ctx, duk_get_top(ctx) - 1);
    cerr << "native_print: " << duk_to_string(ctx, -1) << endl;
    // printf("%s\n", duk_to_string(ctx, -1));
    return 0;
}

// execute the REGISTERED_RUNNER->runEvo()
static duk_ret_t _duktape_run(duk_context * ctx)
{
    if (REGISTERED_RUNNER == 0) return DUK_RET_ERROR;

    REGISTERED_RUNNER->run();
    return 0;
}

duk_context* init_duktape() {
    duk_context* engine = duk_create_heap_default();
    // duk_push_object(engine);
    duk_push_global_object(engine);

    // create the JS-wrapped function setParams() that will call _duktape_setParams()
    duk_push_c_function(engine, _duktape_setParams , 41 );
    duk_put_prop_string(engine, -2 , "setParams");

    // create the JS-wrapped function runEvo() that will call _duktape_run()
    duk_push_c_function(engine, _duktape_run , 0);
    duk_put_prop_string(engine, -2 , "runEvo");

    // // create the JS-wrapped function print() that will call _duktape_print()
    // duk_push_c_function(engine, _duktape_print , 1);
    // duk_put_prop_string(engine, -2 , "print");

    duk_push_c_function(engine, native_print , DUK_VARARGS);
    duk_put_prop_string(engine, -2 , "native_print");

    return engine;
}


void destroy_duktape(duk_context* engine) {
    duk_pop(engine);
    duk_destroy_heap(engine);
}

FuzzyCocoScriptRunner::FuzzyCocoScriptRunner(ScriptRunnerMethod& runner) 
    : _runner(runner)
{
    // register own instance to duktape C function wrappers
    // BEWARE: not thread safe
    REGISTERED_RUNNER = this;
}

FuzzyCocoScriptRunner::~FuzzyCocoScriptRunner() {
    REGISTERED_RUNNER = 0;
}



void FuzzyCocoScriptRunner::evalScriptCode(const string& code) {
    auto engine = init_duktape();

    // Check and evaluate the script
    if (duk_peval_string(engine, code.c_str()) != DUK_EXEC_SUCCESS )
    {
        cerr << "## FuzzyCocoScriptRunner Script error: " <<  duk_safe_to_string(engine , -1) << endl;
        //destroy_duktape(engine);
        throw runtime_error("JS Script error");
    }
    destroy_duktape(engine);
}

void FuzzyCocoScriptRunner::evalSimpleCodeThatReturnsAString(const string& code) {
    auto engine = init_duktape();

    // Check and evaluate the script
    if (duk_peval_string(engine, code.c_str()) != DUK_EXEC_SUCCESS )
    {
        cerr << "##========================================================================\n";
        cerr << "## FuzzyCocoScriptRunner Script error: " <<  duk_safe_to_string(engine , -1) << endl;
        cerr << "## tried to eval code=" << code.c_str() << endl;
        //destroy_duktape(engine);
        throw runtime_error("JS Script error");
    } else {
        cerr << "## evalSimpleCodeThatReturnsAString() returned " << duk_get_string(engine, -1) << endl;
        duk_pop(engine);
    }
    destroy_duktape(engine);
}

void FuzzyCocoScriptRunner::run() {
    cerr << "##========================================================================\n";
    cerr << "##FuzzyCocoScriptRunner::run()\n";
    cerr << "## params: " << _params.coco << endl;
    cerr << "##========================================================================\n";
    _runner.run(_params);
}

void FuzzyCocoScriptRunner::setParams(const ScriptParams& params) {
    _params = params;
    // should check some params
}
