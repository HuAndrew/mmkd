
@MODELS.register_module()
class Fd(nn.Module):
    """PyTorch module for representation distillation using Smooth L1 Loss.

    Args:
        student_num_features (int): Number of output features from the student's feature extractor.
        teacher_num_features (int): Number of output features from the teacher's feature extractor.
        alpha (float): Weighting factor for the distillation loss.
    """

    def __init__(self, student_num_features=256, teacher_num_features=256, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        # Initialize the projector with orthogonal parametrization
        self.projector = nn.utils.parametrizations.orthogonal(
            nn.Linear(student_num_features, teacher_num_features, bias=False)
        )
        # Ensure that the adaptive pooling outputs a feature map of 1x1 size to match fully connected layers
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        """Forward pass of the loss module.

        Args:
            student_features (Tensor): Student features from the forward pass (N, C, H, W).
            teacher_features (Tensor): Teacher features from the forward pass (N, C, H, W).

        Returns:
            Tensor: Computed distillation loss.
        """
        # Pool teacher features over spatial dimensions
        b, c, h, w = teacher_features.size()
        teacher_features_pooled = self.adaptive_avg_pool(teacher_features).view(b, c)

        # Pool student features over spatial dimensions and project
        student_features_pooled = self.adaptive_avg_pool(student_features).view(b, -1)
        student_features_projected = self.projector(student_features_pooled)

        # Apply layer normalization to pooled teacher features
        teacher_features_norm = F.layer_norm(teacher_features_pooled, (teacher_features_pooled.size(1),))

        # Calculate the smooth L1 loss between the projected student features and the normalized teacher features
        distill_loss = self.alpha * F.smooth_l1_loss(student_features_projected, teacher_features_norm)

        return distill_loss
