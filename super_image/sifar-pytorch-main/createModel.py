import newModels
from timm.models import create_model
def create(args):
    if args.model == "nfnet_f3":
        return newModels.NewModel(backBone="nfnet_f3")
    elif args.model == "convnext_base_384_in22ft1k":
        return newModels.NewModel(backBone="convnext_base_384_in22ft1k")
    else:
        return create_model(
        args.model,
        pretrained=args.pretrained,
        duration=args.duration,
        hpe_to_token = args.hpe_to_token,
        rel_pos = args.rel_pos,
        window_size=args.window_size,
        super_img_rows = args.super_img_rows,
        token_mask=not args.no_token_mask,
        online_learning = args.one_w >0.0 or args.dml_w >0.0,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        use_checkpoint=args.use_checkpoint,
        image_size = args.input_size
    )