class TrackableObject:
    def __init__(self, object_id, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.object_id = object_id
        self.centroids = [centroid]
        self.max_size = -1
        self.min_size = -1
        self.stored_size = []
        self.text = 0

        # initialize a boolean used to indicate if the object has
        # already been counted or not
