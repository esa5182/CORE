import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from core_ave import COREave
from core_trm import COREtrm, COREgrt


def run_single_model(args):
    if args.model == 'ave':
        model_class = COREave
    elif args.model == 'trm':
        model_class = COREtrm
    elif args.model == 'grt':
        model_class = COREgrt
    else:
        raise ValueError('model can only be "ave", "trm", or "grt".')

    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=['props/overall.yaml', f'props/core_{args.model}.yaml']
    )

    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    if args.model == 'ave':
        model = COREave(config, train_data.dataset).to(config['device'])
    elif args.model == 'trm':
        model = COREtrm(config, train_data.dataset).to(config['device'])
    elif args.model == 'grt':
        model = COREgrt(config, train_data.dataset).to(config['device'])
    else:
        raise ValueError('model can only be "ave", "trm", or "grt".')

    logger.info(model)

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config['show_progress']
    )

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='trm',
        help='ave or trm or grt'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='diginetica',
        help='diginetica, nowplaying, retailrocket, tmall, yoochoose'
    )
    args, _ = parser.parse_known_args()

    run_single_model(args)
