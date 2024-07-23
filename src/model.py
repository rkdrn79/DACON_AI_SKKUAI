from src.model_factory.deepcnn import DEEPCNN

def get_model(args):
        
    if args.model_name=="resnet101":
        model = DEEPCNN(args)
    else:
        raise ValueError(f"model name {args.model_name} not found")
    
    print(model)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    return model
