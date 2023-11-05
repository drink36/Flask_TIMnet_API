import argparse
import numpy as np
import torch
from timm.models import create_model
import utils
import config
import modeling_finetune
import cv2
import mediapipe as mp
import video_transforms as video_transforms
import volume_transforms as volume_transforms
from PIL import Image


# frame
indices = [1, 3, 5, 7, 9, 11, 12, 14, 16, 18, 19, 21, 23, 25, 27, 29]



def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_dim512_no_depth_patch16_160', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=160, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    parser.add_argument('--attn_type', default='local_global',
                        type=str, help='attention type for spatiotemporal modeling')
    parser.add_argument('--lg_region_size', type=int, nargs='+', default=(2,5,10),
                        help='region size (t,h,w) for local_global attention')
    parser.add_argument('--lg_first_attn_type', type=str, default='self', choices=['cross', 'self'],
                        help='the first attention layer type for local_global attention')
    parser.add_argument('--lg_third_attn_type', type=str, default='cross', choices=['cross', 'self'],
                        help='the third attention layer type for local_global attention')
    parser.add_argument('--lg_attn_param_sharing_first_third', action='store_true',
                        help='share parameters of the first and the third attention layers for local_global attention')
    parser.add_argument('--lg_attn_param_sharing_all', action='store_true',
                        help='share all the parameters of three attention layers for local_global attention')
    parser.add_argument('--lg_classify_token_type', type=str, default='region', choices=['org', 'region', 'all'],
                        help='the token type in final classification for local_global attention')
    parser.add_argument('--lg_no_second', action='store_true',
                        help='no second (inter-region) attention for local_global attention')
    parser.add_argument('--lg_no_third', action='store_true',
                        help='no third (local-global interaction) attention for local_global attention')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=160)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default= 1)
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--data_set', default='FERV39k', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder',
                        'DFEW', 'FERV39k', 'MAFW', 'RAVDESS', 'CREMA-D', 'ENTERFACE'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default = config.MODEL_PATH_VIDEO,
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--depth', default=16, type=int,
                        help='specify model depth, NOTE: only works when no_depth model is used!')


    return parser.parse_args()


def prediction(path):
    args = get_args()
    video = path
    device = torch.device(args.device)

    model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
            depth=args.depth,
            attn_type=args.attn_type,
            lg_region_size=args.lg_region_size, lg_first_attn_type=args.lg_first_attn_type,
            lg_third_attn_type=args.lg_third_attn_type,
            lg_attn_param_sharing_first_third=args.lg_attn_param_sharing_first_third,
            lg_attn_param_sharing_all=args.lg_attn_param_sharing_all,
            lg_classify_token_type=args.lg_classify_token_type,
            lg_no_second=args.lg_no_second, lg_no_third=args.lg_no_third,
    )


    model.to(device)


    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))


    utils.auto_load_model_eval(
        args=args, model=model, model_without_ddp=model_without_ddp,
    )


    ## preprocess
    data_transform = video_transforms.Compose([
        video_transforms.Resize(size=(160, 160), interpolation='bilinear'),
        # me: old, may have bug (heigh != width)
        # video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(160, 160)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    findFace = mp.solutions.face_detection.FaceDetection()
    Faces = []
    Emotions = ['happiness','sadness','neutral','anger','surprise','disgust','fear']
    prediction = 'neutral'
    prediction_list = list()
    posX =1
    posY =1

    cap = cv2.VideoCapture(video)
    #cap.set(cv2.CAP_PROP_FPS, 30)

    model.eval()

    while(True):
        _ , frame = cap.read()
        if frame is None:
            break
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height = frame.shape[0]
        width = frame.shape[1]

        results = findFace.process(frameRGB)

    
        if results.detections != None:
            for face in results.detections:
                bBox = face.location_data.relative_bounding_box
                x,y,w,h = int(bBox.xmin*width),int(bBox.ymin*height),int(bBox.width*width),int(bBox.height*height)
                posX,posY = x,y
                Faces.append((x,y,w,h))
            
        
        if(len(Faces)%30==0):
            images = list()

            for indice in indices:
                Face = Faces[indice]
                x,y,w,h =Face
                faceExp = frame[y:y+h,x:x+w]

                faceExp = cv2.cvtColor(faceExp, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(faceExp)



                images.append(pil)
            
            images = data_transform(images)

            images = images.reshape(-1,3,16,160,160) #[1, 3, 16, 160, 160]

            images = images.to(device, non_blocking=True)
            


            with torch.no_grad():
                output = model(images)
            
            pred = output.argmax(1)
            prediction = Emotions[pred.item()]
            print(prediction)
            prediction_list.append(prediction)
            Faces.clear()
    cap.release()
    cv2.destroyAllWindows()
    return prediction_list