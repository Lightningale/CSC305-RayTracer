import sys
from collections import namedtuple

import numpy as np


THRESHOLD = 0.000_001
MAX_REFLECT = 3


Sphere = namedtuple("Sphere", "position scale color ka kd ks kr n")
Light = namedtuple("Light", "position color")
Resolution = namedtuple("Resolution", "width height")


lights = []
spheres = []
ambient = None
near_plane = {}


def normalize(vector_array):
	return vector_array * (1.0 / np.sqrt(np.sum(vector_array**2, axis=-1)))[..., np.newaxis]


def reflect_across(v, n):
	proj = n * np.sum(n * v, axis=-1)[..., np.newaxis]
	return 2 * proj - v


def get_ds(ray_directions, sphere, intersection_point, intersection_normal, light):
	shadow_ray = normalize(light.position - intersection_point)
	object_intersection_metric = trace(intersection_point, shadow_ray, update_nothing, depth=0)
	light_metric = np.sum((light.position - intersection_point)**2, axis=-1)

	#diffuse and specular with shadows
	diffuse_product = np.sum(intersection_normal * shadow_ray, axis=-1)
	diffuse = sphere.kd * sphere.color * light.color * np.maximum(0, diffuse_product)[..., np.newaxis]
	r = reflect_across(shadow_ray, intersection_normal)
	specular = light.color * sphere.ks * np.power(np.maximum(0, np.sum(-r * normalize(ray_directions), axis=-1)), sphere.n)[..., np.newaxis]
	return (~(light_metric > object_intersection_metric))[..., np.newaxis] * (diffuse + specular)


def trace(ray_positions, ray_directions, update, depth):
	first_intersection_metric = np.full(ray_positions.shape[:-1], np.nan)
	first_intersection_point = np.full(ray_positions.shape, np.nan)
	first_intersection_normal = first_intersection_point.copy()
	first_intersection_sphere = np.zeros(ray_positions.shape[:-1], dtype=np.int8)
	
	#calculate ray sphere intersection
	for sphere_index, sphere in enumerate(spheres, start=1):
		local_p = (ray_positions - sphere.position) / sphere.scale
		local_d = ray_directions / sphere.scale
		
		# Solve (px + t dx)^2 + (py + t dy)^2 + (pz + t dz)^2 = 1 for t
		#       px^2 + 2t px dx + t^2 dx^2 + ... = 1
		#       |p|^2 + 2t(p * d) + t^2 |d|^2 = 1
		#       t = (-b +- sqrt(b^2 - 4ac))/2a, a = |d|^2, b = 2 p*d, c = |p|^2-1
		a = np.sum(local_d**2, axis=-1)
		inv_2a = 0.5 / a
		b = 2 * np.sum(local_p * local_d, axis=-1)
		c = np.sum(local_p**2, axis=-1) - 1.0
		discriminant = b**2 - 4*a*c
		has_intersection = discriminant >= 0

		rd = np.zeros_like(discriminant)
		rd[has_intersection] = np.sqrt(discriminant[has_intersection])

		t0 = np.where(has_intersection, (-b - rd) * inv_2a, np.nan)
		t1 = np.where(has_intersection, (-b + rd) * inv_2a, np.nan)

		t = np.where(t0 >= THRESHOLD, t0,
					np.where(t1 >= THRESHOLD, t1, np.full(t1.shape, np.nan)))

		local_ip = local_p + t[..., np.newaxis] * local_d
		local_in = np.where((np.sum(local_ip * local_d, axis=-1) > 0)[..., np.newaxis], -local_ip, local_ip)  # flip normals on the inside

		intersection_point = local_ip * sphere.scale + sphere.position
		intersection_normal = normalize(local_in / sphere.scale)
		intersection_metric = np.sum((intersection_point - ray_positions)**2, axis=-1)

		#calculate clipping
		closest = ~np.isnan(intersection_metric) & ~(intersection_metric >= first_intersection_metric)
		first_intersection_metric[closest] = intersection_metric[closest]
		first_intersection_normal[closest] = intersection_normal[closest]
		first_intersection_point[closest] = intersection_point[closest]
		first_intersection_sphere[closest] = sphere_index

	for sphere_index, sphere in enumerate(spheres, start=1):
		nz = (first_intersection_sphere == sphere_index).nonzero()
		update(sphere, nz, first_intersection_point[nz], first_intersection_normal[nz], ray_directions[nz], depth=depth)

	return first_intersection_metric


def update_nothing(*args, **kwargs):
	pass


def update_illumination(illumination):
	def f(sphere, locs, intersection_point, intersection_normal, ray_directions, depth):
		adsr = sphere.ka * sphere.color * ambient + np.sum([get_ds(ray_directions, sphere, intersection_point, intersection_normal, light) for light in lights], axis=0)

		if depth < MAX_REFLECT:
			reflect_illumination = np.zeros(ray_directions.shape[:-1] + (3,))
			trace(intersection_point, reflect_across(-ray_directions, intersection_normal), update_illumination(reflect_illumination), depth=depth + 1)
			adsr += sphere.kr * reflect_illumination

		illumination[locs] = adsr

	return f


def main():
	global ambient

	ambient = background = resolution = None
	output="default.ppm"
	with open(sys.argv[1], "r") as f:
		for line in f:
			words = line.split()

			match words:
				case [("NEAR" | "LEFT" | "RIGHT" | "BOTTOM" | "TOP") as prop, w]: near_plane[prop] = float(w)
				case ["RES", w, h]: resolution = Resolution(int(w), int(h))
				case ["AMBIENT", r, g, b]: ambient = np.array([*map(float, (r, g, b))])
				case ["BACK", r, g, b]: background = np.array([*map(float, (r, g, b))])
				case ["SPHERE", name, *args]:
					x, y, z, sx, sy, sz, r, g, b, *lighting = map(float, args)
					spheres.append(Sphere(np.array([x, y, z]), np.array([sx, sy, sz]), np.array([r, g, b]), *lighting))
				case ["LIGHT", name, *args]:
					x, y, z, r, g, b = map(float, args)
					lights.append(Light(np.array([x, y, z]), np.array([r, g, b])))
				case ["OUTPUT", name]:
					output=name
				case _: print(f"Warning: ignoring unrecognized directive {line.rstrip()!r}", file=sys.stderr)

	#stack rays for matrix calculation
	ray_directions = np.dstack(np.meshgrid(
		np.linspace(near_plane["LEFT"], near_plane["RIGHT"], resolution.width, endpoint=False),
		np.linspace(near_plane["TOP"], near_plane["BOTTOM"], resolution.height, endpoint=False),
	) + [-near_plane["NEAR"] * np.ones((resolution.width, resolution.height))]).reshape((resolution.width * resolution.height, 3))

	ray_positions = ray_directions

	illumination = np.full((resolution.width * resolution.height, 3), background)
	_ = trace(ray_positions, ray_directions, update_illumination(illumination), depth=0)
	image = illumination.clip(0.0, 1.0).reshape(resolution + (3,))


	#from matplotlib import pyplot as plt
	#plt.imshow(image)
	#plt.show()

	with open(output, "wb") as out:
		out.write(b"P6\n%d %d\n255\n" % resolution)
		out.write((image * 255).astype(np.uint8).tobytes())


if __name__ == "__main__":
	main()
