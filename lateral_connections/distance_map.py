from abc import ABC
import numpy as np
import torch

class DistanceMapTemplate(ABC):
    @classmethod
    def build_distance_map_template(cls, feature_maps, pixel_radius):
        raise NotImplementedError()

    # Takes a feature map (N, Height, Width) and returns a (Height1, Width1, Height2, Width2)
    # 4-dimensional distance map where the height2/width2 represent the distance to other pixels
    # in the same feature map and height1/width1 correspond to the pixel on which it is centered
    # around. The given function can be exchanged to provide different distances, default
    # is the manhattan distance.
    #
    @classmethod
    def generate_distance_map(cls, feature_maps, pixel_radius):
        distance_template = cls.build_distance_map_template(feature_maps, pixel_radius=pixel_radius)

        distance_map = np.zeros(feature_maps.shape[1:] + feature_maps.shape[1:])
        center_pos_x = distance_template.shape[0] // 2
        center_pos_y = distance_template.shape[1] // 2

        for x in range(feature_maps.shape[1]):
            for y in range(feature_maps.shape[2]):
                template_x = center_pos_x - x
                template_y = center_pos_y - y

                distance_map[x, y, :, :] = distance_template[template_x:template_x+feature_maps.shape[1], template_y:template_y+feature_maps.shape[2]]
                distance_map[x, y, :, :] /= np.sum(distance_map[x, y, :, :])
                
        return distance_map

class ManhattanDistanceTemplate(DistanceMapTemplate):
    # Generates a template that is of almost double both dimensions
    # of the provided feature map and puts the maximum in the center.
    #
    # Can be used to extract parts of the pattern for further processing
    #
    @classmethod
    def build_distance_map_template(cls, feature_maps, pixel_radius):
        feature_map_dimension_x = 2 * feature_maps.shape[1] - 1
        feature_map_dimension_y = 2 * feature_maps.shape[2] - 1

        distance_template = torch.ones((feature_map_dimension_x, feature_map_dimension_y)).float()
        distance_template = torch.mul(pixel_radius, distance_template)
        
        center_pos_x = feature_map_dimension_x // 2
        center_pos_y = feature_map_dimension_y // 2

        for x in range(feature_map_dimension_x):
            for y in range(feature_map_dimension_y):
                x_dist = np.abs(x - center_pos_x)
                y_dist = np.abs(y - center_pos_y)
                
                if torch.le(torch.add(x_dist, y_dist), pixel_radius):
                    distance_template[x, y] = torch.add(x_dist, y_dist)


        distance_template = distance_template.cpu().numpy()
        distance_template = np.divide(np.subtract(np.max(distance_template), distance_template), np.max(distance_template))
        # distance_template[np.where(distance_template > 0)] = 1

        return distance_template
