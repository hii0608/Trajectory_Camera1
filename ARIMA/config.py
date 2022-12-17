import argparse

def arima_parse(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    enc_parser.set_defaults(order=(2,1,2),
                            freq='D',
                            path = 'output/result.csv',
                            # model_save = 'Final/Energy_pv_arima/model/ARIMA_fit.pkl',
                            model_save = 'model/ARIMA_fit_x.pt',
                            date_column = 'time',  #해당값에는 data_column이 없는데.. resampling 안 할 거니까 괜찮지 않나???
                            pred_result='output/arima_predict_x.csv',
                            result_visualize='output/arima_predict_x.png',
                            )    # True

    # return enc_parser.parse_args(arg_str)