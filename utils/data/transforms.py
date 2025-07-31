class ClassInfoMixin:
    @property
    def classes(self):
        return list(range(self.num_classes))

