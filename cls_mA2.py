from cls_Camera import Camera

class mA2:    

    def __init__(self,
                 camera_fp_f: Camera,
                 camera_fs_f: Camera,
                 camera_fs_s: Camera,
                 camera_ap_p: Camera,
                 camera_ap_a: Camera,
                 camera_as_f: Camera,
                 camera_as_s: Camera):
        self._fp_f = camera_fp_f
        self._fs_f = camera_fs_f
        self._fs_s = camera_fs_s
        self._ap_p = camera_ap_p
        self._ap_a = camera_ap_a
        self._as_a = camera_as_f
        self._as_s = camera_as_s
    
    @property
    def fp_f(self):
        return self._fp_f
    @property
    def fs_f(self):
        return self._fs_f
    @property
    def fs_s(self):
        return self._fs_s
    @property
    def ap_p(self):
        return self._ap_p
    @property
    def ap_a(self):
        return self._ap_a
    @property
    def as_a(self):
        return self._as_a
    @property
    def as_s(self):
        return self._as_s