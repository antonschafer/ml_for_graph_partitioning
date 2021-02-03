import torch
import torch.nn as nn


class ComposedModel(nn.Module):
    def __init__(self, models, train_models, save_fnames, load_fnames, parallel):
        super(ComposedModel, self).__init__()

        self.train_models = train_models
        self.training = False
        self.save_fnames = save_fnames
        self.load_fnames = load_fnames

        # use multiple GPUs
        self.parallel = parallel
        if parallel:
            models = [nn.DataParallel(m) for m in models]

        self.models = nn.ModuleList(models)
        self.load()

    def train(self, mode=True):
        """Sets the module in training mode."""
        self.training = mode
        for model, train in zip(self.models, self.train_models):
            if train:
                model.train(mode)
            else:
                model.eval()
        return self

    def training_params(self):
        params = []
        for model, train in zip(self.models, self.train_models):
            if train:
                params += [p for p in model.parameters() if p.requires_grad]
        return params

    def named_training_params(self):
        params = []
        for model, train in zip(self.models, self.train_models):
            if train:
                if self.parallel:
                    model_name = model.module.__class__.__name__
                else:
                    model_name = model.__class__.__name__
                params += [
                    (model_name + (name[7:] if self.parallel else name), p)
                    for name, p in model.named_parameters()
                    if p.requires_grad
                ]
        return params

    def training_params_by_model(self):
        params = {}
        for model, train in zip(self.models, self.train_models):
            if train:
                if model.__class__.__name__ != "DataParallel":
                    model_name = model.__class__.__name__
                else:
                    model_name = model.module.__class__.__name__
                params[model_name] = [p for p in model.parameters() if p.requires_grad]
        return params

    def num_params(self):
        return sum(p.numel() for p in self.training_params())

    def save(self, choice=None):
        if choice is None:
            choice = [True for _ in self.models]
        for save, model, fname in zip(choice, self.models, self.save_fnames):
            if save:
                if self.parallel:
                    torch.save(model.module.state_dict(), fname)
                else:
                    torch.save(model.state_dict(), fname)

    def load(self, fnames=None):
        if fnames is None:
            fnames = self.load_fnames

        assert len(fnames) == len(self.models), "Wrong number of submodel files given"
        for model, fname in zip(self.models, fnames):
            if fname is not None:
                if self.parallel:
                    model = model.module
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(fname))
                else:
                    model.load_state_dict(torch.load(fname, map_location="cpu"))
