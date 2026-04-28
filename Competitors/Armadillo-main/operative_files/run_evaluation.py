from Code._embed_all_no_paral import run_experiment
from Code._overlap_of_a_pair_effe_effi_processing import recompute_embeddings_overlaps_overlap_computation_time, repeat_test_emb_already_computed, add_new_column_prediction_armadillo
from Code._performance_overlap_computation import *
from Code._table_stats_computation import compute_tables_stats
from Code._table_querying import *
import os
from pathlib import Path

def evaluate(model_GitTables: str=None, table_dict_GitTables: str | dict=None, model_WikiLast: str=None, table_dict_WikiLast: str | dict=None, 
             gittables_unlabeled: str=None, wikilast_unlabeled: str=None, chart_data_path: str=None, charts_path: str=None) -> None:
    #prepare data for charts
    #Embedding generation time
    print('Test: embedding all of the tables in GitTables using armadillo_gittables')
    run_experiment(model_file=model_GitTables, table_dict_path=table_dict_GitTables, iters=3, experiment_data_file_path=chart_data_path+'/embedding_gen_time_gittables.pkl', embedding_file=chart_data_path+'/embedding_file_git_on_git.pkl')

    # print('Test: embedding all of the tables in GitTables using armadillo_wikilast')
    run_experiment(model_file=model_WikiLast, table_dict_path=table_dict_GitTables, iters=1, embedding_file=chart_data_path+'/embedding_file_wiki_on_git.pkl')
    
    print('Test: embedding all of the tables in WikiLast using armadillo_wikilast')
    run_experiment(model_file=model_WikiLast, table_dict_path=table_dict_WikiLast, iters=3, experiment_data_file_path=chart_data_path+'/embedding_gen_time_wikilast.pkl', embedding_file=chart_data_path+'/embedding_file_wiki_on_wiki.pkl')
    
    # print('Test: embedding all of the tables in WikiLast using armadillo_gittables')
    run_experiment(model_file=model_GitTables, table_dict_path=table_dict_WikiLast, iters=1, embedding_file=chart_data_path+'/embedding_file_git_on_wiki.pkl')

    #Major comparison
    recompute_embeddings_overlaps_overlap_computation_time(unlabeled_dataset=gittables_unlabeled, model_file=model_GitTables, table_dict=table_dict_GitTables, output_file=chart_data_path+'/effe_effi_gittables.csv')
    repeat_test_emb_already_computed(old_file=chart_data_path+'/effe_effi_gittables.csv', embeddings_dict=chart_data_path+'/embedding_file_git_on_git.pkl', out_path=chart_data_path+'/effe_effi_gittables.csv')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_gittables.csv', embedding_dict=chart_data_path+'/embedding_file_git_on_git.pkl', out_path=chart_data_path+'/effe_effi_gittables.csv', label='gittables')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_gittables.csv', embedding_dict=chart_data_path+'/embedding_file_wiki_on_git.pkl', out_path=chart_data_path+'/effe_effi_gittables.csv', label='wikilast')


    recompute_embeddings_overlaps_overlap_computation_time(unlabeled_dataset=wikilast_unlabeled, model_file=model_WikiLast, table_dict=table_dict_WikiLast, output_file=chart_data_path+'/effe_effi_wikilast.csv')
    repeat_test_emb_already_computed(old_file=chart_data_path+'/effe_effi_wikilast.csv', embeddings_dict=chart_data_path+'/embedding_file_wiki_on_wiki.pkl', out_path=chart_data_path+'/effe_effi_wikilast.csv')

    # add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_wikilast.csv', embedding_dict=chart_data_path+'/embedding_file_git_on_wiki.pkl', out_path=chart_data_path+'/effe_effi_wikilast.csv', label='gittables')
    add_new_column_prediction_armadillo(old_data=chart_data_path+'/effe_effi_wikilast.csv', embedding_dict=chart_data_path+'/embedding_file_wiki_on_wiki.pkl', out_path=chart_data_path+'/effe_effi_wikilast.csv', label='wikilast')


    #NDCG table querying
    run_table_querying_experiment(query_set=root_gittables+'/query_set_1k.pkl', target_set=root_gittables+'/data_lake_10k.pkl', model=model_GitTables, target_embedding_dict=chart_data_path+'/embedding_file_git_on_git.pkl', 
                                table_dict=table_dict_GitTables, outpath=chart_data_path+'/table_querying_arm.pkl')
    add_armadillo_predictions_to_sloth_file(sloth_baseline_out='/home/francesco.pugnaloni/tmp/gittables_root/table_querying.csv', armadillo_out=chart_data_path+'/table_querying_arm.pkl', outpath=chart_data_path+'/armadillo_sloth_table_querying_enriched.csv')
    compute_ndcg_at_k(table_querying_arm_sloth=chart_data_path+'/armadillo_sloth_table_querying_enriched.csv', query_set=root_gittables+'/query_set_1k.pkl', outpath=chart_data_path+'/ndcg_score.csv')


    # plot chart MAE compared to baseline
    print('Generating MAE boxplot for GitTables')
    plot_box_group(chart_data_path+'/effe_effi_gittables.csv', label_list=['armadillo_gittables_AE', 'armadillo_wikilast_AE','AE_os_sim','AE_jsim'], out_pdf=charts_path+'/mae_comparison_baseline_gittables.pdf')
    plot_box_group(chart_data_path+'/effe_effi_gittables.csv', label_list=['armadillo_gittables_AE','AE_os_sim','AE_jsim'], out_pdf=charts_path+'/mae_comparison_baseline_gittables.pdf')
    
    print('Generating MAE boxplot for WikiLast')
    plot_box_group(chart_data_path+'/effe_effi_wikilast.csv', label_list=['armadillo_wikilast_AE','armadillo_gittables_AE','AE_os_sim','AE_jsim'],out_pdf=charts_path+'/mae_comparison_baseline_wikilast.pdf')
    plot_box_group(chart_data_path+'/effe_effi_wikilast.csv', label_list=['armadillo_wikilast_AE','AE_os_sim','AE_jsim'],out_pdf=charts_path+'/mae_comparison_baseline_wikilast.pdf')
    
    #plot chart MAE per bin compared to baseline
    print('Generating MAE per bin barplot for GitTables')
    compare_models_hist(chart_data_path+'/effe_effi_gittables.csv', out_pdf=charts_path+'/mae_comparison_baseline_per_bin_gittables.pdf', font_scale=1.1)
    
    print('Generating MAE per bin barplot for WikiLast')
    compare_models_hist(chart_data_path+'/effe_effi_wikilast.csv', out_pdf=charts_path+'/mae_comparison_baseline_per_bin_wikilast.pdf', font_scale=1.1)

    #plot chart NDCG
    print('Generating ndcg barplot')
    compare_models_ndcg(chart_data_path+'/ndcg_score.csv', font_scale=1.1, out_pdf=charts_path+'/ndcg_score.pdf')

    #plot chart overlap computation time
    print('Generating scatterplot for comparing overlap computation time on GitTables')
    show_scatter_t_exec_sloth_arm(chart_data_path+'/effe_effi_gittables.csv', logx=True, logy=True, output_pdf=charts_path+'/overlap_pair_t_exc_comparison_arm_sloth_gittabels.pdf')
    print('Generating scatterplot for comparing overlap computation time on WikiLast')
    show_scatter_t_exec_sloth_arm(chart_data_path+'/effe_effi_wikilast.csv', logx=True, logy=True, output_pdf=charts_path+'/overlap_pair_t_exc_comparison_arm_sloth_wikilast.pdf')

    #plot chart embedding generation time with increasing table areas
    print('Generating scatterplot to show how embedding generation time varies with increasing table areas on GitTables')
    visualize_scatter_plot(chart_data_path+'/embedding_gen_time_gittables.pkl', logx=True, logy=True, out_pdf=charts_path+'/emb_gen_t_exec_gittables.pdf', font_size=15)
    print('Generating scatterplot to show how embedding generation time varies with increasing table areas on WikiLast')
    visualize_scatter_plot(chart_data_path+'/embedding_gen_time_wikilast.pkl', logx=True, logy=True, out_pdf=charts_path+'/emb_gen_t_exec_wikitables.pdf', font_size=15)

    #compute table stats
    print('Computing table stats for GitTables')
    compute_tables_stats(table_dict_GitTables, root_gittables+'/table_stats.csv')
    print('Computing table stats for WikiLast')
    compute_tables_stats(table_dict_WikiLast, root_wikilast+'/table_stats.csv')

if __name__ == '__main__':
    root = '/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/charts/'               # Insert the full name of the root directory here
    root_gittables = root + ''
    root_wikilast = root

    params = {
        'model_GitTables': None, 
        'table_dict_GitTables': None, 
        'model_WikiLast': None, 
        'table_dict_WikiLast': None, 
        'gittables_unlabeled': None, 
        'wikilast_unlabeled': None, 
        'chart_data_path': None, 
        'charts_path': None
    }

    evaluate(
        model_WikiLast='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/models/armadillo_wiki.pth',
        table_dict_WikiLast='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/dictionaries/table_dict.pkl',
        chart_data_path='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/charts/tmp',
        charts_path='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/charts/tmp',
        wikilast_unlabeled='/home/francesco.pugnaloni/armadillo_all/datasets/WikiTables/test.csv'
    )