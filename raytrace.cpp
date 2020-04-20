//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

const mat4 Cylinder = mat4 (vec4(1,0,0,0),
			                vec4(0,1,0,0),
			                vec4(0,0,0,0),
			                vec4(0,0,0,-1));

const mat4 Sphere = mat4 (vec4(1,0,0,0),
					      vec4(0,1,0,0),
					      vec4(0,0,1,0),
					      vec4(0,0,0,-1));

const mat4 Ellipsoid = mat4 (vec4(1,0,0,0),
					         vec4(0,1,0,0),
					         vec4(0,0,1,0),
					         vec4(0,0,0,-1));

const mat4 Paraboloid = mat4 (vec4(1,0,0,0),
							 vec4(0,1,0,0),
							 vec4(0,0,0,0.5),
							 vec4(0,0,0.5, 0));

const mat4 Hyperboloid = mat4 (vec4(1,0,0,0),
							  vec4(0,1,0,0),
							  vec4(0,0,-1,0),
							  vec4(0,0,0, -1));

enum MaterialType{ ROUGH, REFLECTIVE };

mat4 invert(const mat4 r){
	return mat4(vec4((1 / r[0][0]), 0, 0, 0),
				vec4(0, (1 / r[1][1]), 0, 0),
				vec4(0, 0, (1 / r[2][2]), 0),
				vec4(0, 0, 0, 1));
}

mat4 transpose(const mat4 r){
	mat4 ret;
	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 4; j++)
			ret[j][i] = r[i][j];

	return ret;
}

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t;}
};

struct RoughMaterial : Material{
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd= _kd;
		ks = _ks;
		shininess = _shininess;
	}
};
vec3 operator/(vec3 num, vec3 denom){
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}
struct ReflectiveMaterial : Material{
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE){
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};



class Quadratic : public Intersectable{
	mat4 Q;
	vec3 center;			//TODO forgatás transzformáció
	vec3 radius;


protected:
	float f(vec4 r){
		return dot(r * Q, r);
	}

	float f2(vec4 r, vec4 n){
		return dot(r * Q, n);
	}

	vec3 gradf(vec4 r){
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}



public:

	Quadratic(const vec3& _center, const vec3& _radius, Material* _material, mat4 q){
		center = _center;
		radius = _radius;
		material = _material;


		Q = invert(ScaleMatrix(radius)) * q * invert(ScaleMatrix(radius));
		Q = TranslateMatrix(center) * Q * transpose(TranslateMatrix(center));

	}

	Hit intersect(const Ray& ray){


		vec4 start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 direction = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		Hit hit;

		float a = f(direction);
		float b = f2(direction, start) + f2(start, direction);
		float c = f(start);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0)
			return hit;
		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 r = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 g = gradf(r);
		hit.normal = normalize(vec3(g.x, g.y, g.z));
		hit.material = material;
		return hit;

	}
};

class Cylinder : public Quadratic{
	mat4 Q;
	vec3 center;			//TODO forgatás transzformáció
	vec3 radius;
	vec3 cutterPlane1;
	vec3 cutterPlane2;

	vec3 normalVector(vec3 plane){
		vec3 P1 = vec3(3,4, plane.z);
		vec3 P2 = vec3(5,6, plane.z);
		vec3 P3= vec3(6,3, plane.z);

		return cross(vec3(P2.x - P1.x, P2.y - P1.y, P2.z - P1.z), vec3(P3.x - P1.x, P3.y - P1.y, P3.z - P1.z));

	}

	float planeEquation(vec3 t, vec3 plane){
		vec3 p = vec3(3,4, plane.z);

		vec3 n = normalVector(plane);

		return dot(vec3(p.x - t.x, p.y - t.y, p.z - t.z), n);
	}


public:
	Cylinder(const vec3& _center, const vec3& _radius, Material* _material, mat4 q, const vec3& _p1, const vec3& _p2): Quadratic(_center, _radius, _material, q) {
		center = _center;
		radius = _radius;
		material = _material;
		cutterPlane1 = _p1;
		cutterPlane2 = _p2;

		mat4 M = mat4(vec4(radius.x, 0, 0, 0),
					  vec4(0, radius.y,0,0),
					  vec4(0,0, radius.z,0),
					  vec4(0,0,0,1));

		Q = invert(M) * q * invert(M);
		Q = TranslateMatrix(center) * Q * transpose(TranslateMatrix(center));
	}

	Hit intersect(const Ray& ray){


		vec4 start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 direction = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		Hit hit;

		float a = f(direction);
		float b = f2(direction, start) + f2(start, direction);
		float c = f(start);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		float p1 = planeEquation(ray.start + ray.dir * t1, cutterPlane1);
		float p2 = planeEquation(ray.start + ray.dir * t1, cutterPlane2);

		if(!(p1 < 0 && p2 > 0)) t1 = -1;

		float f1 = planeEquation(ray.start + ray.dir * t2, cutterPlane1);
		float f2 = planeEquation(ray.start + ray.dir * t2, cutterPlane2);

		if(!(f1 < 0 && f2 > 0)) t2 = -1;


		if (t1 <= 0 && t2 <= 0)
			return hit;

		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 r = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 g = gradf(r);
		hit.normal = normalize(vec3(g.x, g.y, g.z));
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		hit.material = material;
		return hit;

	}

};

class Hyperboloid : public Intersectable{
	mat4 Q;
	vec3 center;			//TODO forgatás transzformáció
	vec3 radius;
	vec3 cutterPlane1;
	vec3 cutterPlane2;

	float f(vec4 r){
		return dot(r * Q, r);
	}

	float f2(vec4 r, vec4 n){
		return dot(r * Q, n);
	}

	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);

	}
		vec3 normalVector(vec3 plane){
		vec3 P1 = vec3(3,4, plane.z);
		vec3 P2 = vec3(5,6, plane.z);
		vec3 P3= vec3(6,3, plane.z);

		return cross(vec3(P2.x - P1.x, P2.y - P1.y, P2.z - P1.z), vec3(P3.x - P1.x, P3.y - P1.y, P3.z - P1.z));

	}

	float planeEquation(vec3 t, vec3 plane){
		vec3 p = vec3(3,4, plane.z);

		vec3 n = normalVector(plane);

		return dot(vec3(p.x - t.x, p.y - t.y, p.z - t.z), n);
	}


public:
	Hyperboloid(const vec3& _center, const vec3& _radius, Material* _material, mat4 q, const vec3& _p1, const vec3& _p2){
		center = _center;
		radius = _radius;
		material = _material;
		cutterPlane1 = _p1;
		cutterPlane2 = _p2;

		mat4 M = mat4(vec4(radius.x, 0, 0, 0),
					  vec4(0, radius.y,0,0),
					  vec4(0,0, radius.z,0),
					  vec4(0,0,0,1));

		Q = q;
		Q = invert(ScaleMatrix(radius)) * q * invert(ScaleMatrix(radius));
		Q = TranslateMatrix(center) * Q * transpose(TranslateMatrix(center));
	}

	Hit intersect(const Ray& ray){


		vec4 start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 direction = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		Hit hit;

		float a = f(direction);
		float b = f2(direction, start) + f2(start, direction);
		float c = f(start);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		float p1 = planeEquation(ray.start + ray.dir * t1, cutterPlane1);
		float p2 = planeEquation(ray.start + ray.dir * t1, cutterPlane2);

		if(!(p1 < 0 && p2 > 0)) t1 = -1;

		float f1 = planeEquation(ray.start + ray.dir * t2, cutterPlane1);
		float f2 = planeEquation(ray.start + ray.dir * t2, cutterPlane2);

		if(!(f1 < 0 && f2 > 0)) t2 = -1;


		if (t1 <= 0 && t2 <= 0)
			return hit;

		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 r = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 g = gradf(r);
		hit.normal = normalize(vec3(g.x, g.y, g.z));
		if (dot(hit.normal, ray.dir) < 0) hit.normal = hit.normal * (-1); // flip the normal, we are inside the sphere
		hit.material = material;
		return hit;

	}

};
class Paraboloid : public Quadratic{
	mat4 Q;
	vec3 center;			//TODO forgatás transzformáció
	vec3 radius;
	vec3 cutterPlane1;

	vec3 normalVector(vec3 plane){
		vec3 P1 = vec3(3,4, plane.z);
		vec3 P2 = vec3(5,6, plane.z);
		vec3 P3= vec3(6,3, plane.z);

		return cross(vec3(P2.x - P1.x, P2.y - P1.y, P2.z - P1.z), vec3(P3.x - P1.x, P3.y - P1.y, P3.z - P1.z));

	}

	float planeEquation(vec3 t, vec3 plane){
		vec3 p = vec3(3,4, plane.z);

		vec3 n = normalVector(plane);

		return dot(vec3(p.x - t.x, p.y - t.y, p.z - t.z), n);
	}


public:
	Paraboloid(const vec3& _center, const vec3& _radius, Material* _material, mat4 q, const vec3& _p1): Quadratic(_center, _radius, _material, q) {
		center = _center;
		radius = _radius;
		material = _material;
		cutterPlane1 = _p1;;

		mat4 M = mat4(vec4(radius.x, 0, 0, 0),
					  vec4(0, radius.y,0,0),
					  vec4(0,0, radius.z,0),
					  vec4(0,0,0,1));

		Q = invert(M) * q * invert(M);
		Q = TranslateMatrix(center) * Q * transpose(TranslateMatrix(center));
	}

	Hit intersect(const Ray& ray) override{


		vec4 start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 direction = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		Hit hit;

		float a = f(direction);
		float b = f2(direction, start) + f2(start, direction);
		float c = f(start);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		float p1 = planeEquation(ray.start + ray.dir * t1, cutterPlane1);

		if((p1 < 0)) t1 = -1;

		float f1 = planeEquation(ray.start + ray.dir * t2, cutterPlane1);

		if((f1 < 0 )) t2 = -1;


		if (t1 <= 0 && t2 <= 0)
			return hit;

		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 r = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 g = gradf(r);
		hit.normal = normalize(vec3(g.x, g.y, g.z));
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		hit.material = material;
		return hit;

	}

};

class Room : public Quadratic{
	mat4 Q;
	vec3 center;
	vec3 radius;
	vec3 cutterPlane1;

	vec3 normalVector(vec3 plane){
		vec3 P1 = vec3(3,4, plane.z);
		vec3 P2 = vec3(5,6, plane.z);
		vec3 P3= vec3(6,3, plane.z);

		return cross(vec3(P2.x - P1.x, P2.y - P1.y, P2.z - P1.z), vec3(P3.x - P1.x, P3.y - P1.y, P3.z - P1.z));

	}

	float planeEquation(vec3 t, vec3 plane){
		vec3 p = vec3(3,4, plane.z);

		vec3 n = normalVector(plane);

		return dot(vec3(p.x - t.x, p.y - t.y, p.z - t.z), n);
	}


public:
	Room(const vec3& _center, const vec3& _radius, Material* _material, mat4 q, const vec3& _p1): Quadratic(_center, _radius, _material, q) {
		center = _center;
		radius = _radius;
		material = _material;
		cutterPlane1 = _p1;

		mat4 M = mat4(vec4(radius.x, 0, 0, 0),
					  vec4(0, radius.y,0,0),
					  vec4(0,0, radius.z,0),
					  vec4(0,0,0,1));

		Q = invert(M) * q * invert(M);
		Q = TranslateMatrix(center) * Q * transpose(TranslateMatrix(center));
	}

	Hit intersect(const Ray& ray){


		vec4 start = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 direction = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		Hit hit;

		float a = f(direction);
		float b = f2(direction, start) + f2(start, direction);
		float c = f(start);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;

		float p1 = planeEquation(ray.start + ray.dir * t1, cutterPlane1);


		if((p1 > 0)) t1 = -1;

		float f1 = planeEquation(ray.start + ray.dir * t2, cutterPlane1);

		if((f1 > 0 )) t2 = -1;

		if (t1 <= 0 && t2 <= 0)
			return hit;

		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec4 r = vec4(hit.position.x, hit.position.y, hit.position.z, 1);
		vec3 g = gradf(r);
		hit.normal = normalize(vec3(g.x, g.y, g.z));
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		hit.material = material;
		return hit;

	}

};


class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;

	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};


float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La, Le;
	vec3 lightDirection;
	std::vector<vec3> controlPoints;

public:
	void build() {

		vec3 eye = vec3(0,1.8, 0.4), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.6f);
		Le = vec3(2, 2, 2);
		lightDirection = vec3(4, 4, 4);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.3f, 0.2f, 1.0f), ks(1, 1, 1);
		vec3 kd2(0.4f, 0.25f, 0.1f), ks2(1, 1, 1);
		vec3 kd3(0.8f, 0.1f, 0.1f), ks3(1, 1, 1);

		Material * material1 = new RoughMaterial(kd, ks, 50);
		Material * material2 = new RoughMaterial(kd2, ks2, 50);
		Material * material3 = new RoughMaterial(kd3, ks3, 50);

		vec3 n(0.17, 0.35, 1.5), kappa(3.1, 2.7, 1.9);
		Material * gold = new ReflectiveMaterial(n, kappa);

		vec3 n2(0.14, 0.16, 0.13), kappa2(4.1, 2.3, 3.1);
		Material * silver = new ReflectiveMaterial(n2, kappa2);
		objects.push_back(new class Hyperboloid(vec3(0.0f, 0.0f, -0.95f),  vec3(0.6244f, 0.6244f, 1.0f), silver, Hyperboloid, vec3(0.0f, 0.0f, 3.0f), vec3(0.0f, 0.0f, 0.95f)));
		objects.push_back(new class Room(vec3(0.0f, 0.0f, 0.0f),  vec3(2.0f, 2.0f, 1.0f), material2, Ellipsoid, vec3(0,0, 0.95)));


		objects.push_back(new class Cylinder(vec3(0.6f, 0.7f, -0.2f),  vec3(0.15f, 0.15f, 0.15f), material1, Cylinder, vec3(0,0, 0.37),vec3(0,0, -0.5)));
		objects.push_back(new class Cylinder(vec3(-0.7f, -0.8f, -0.4f),  vec3(0.15f, 0.15f, 0.15f), material1, Cylinder, vec3(0,0, 0.35),vec3(0,0, -0.5)));

		objects.push_back(new class Paraboloid(vec3(0.0f, 0.0f, -0.5f),  vec3(0.6f, 0.6f, 1.0f), gold, Paraboloid, vec3(0,0, -0.5)));

		objects.push_back(new class Quadratic(vec3(0.42f, -0.65f, 0.5f),  vec3(0.4f, 0.2f, 0.2f), material3, Ellipsoid));




		for(int i = 0; i < 15; i++){
			controlPoints.push_back(vec3(rnd() * 0.62f, rnd() * 0.62f, 0.95));
			controlPoints.push_back(vec3(-rnd() * 0.62f, rnd() * 0.62f, 0.95));
			controlPoints.push_back(vec3(rnd() * 0.62f, -rnd() * 0.62f, 0.95));
			controlPoints.push_back(vec3(-rnd() * 0.62f, -rnd() * 0.62f, 0.95));
		}
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {    // for directional lights
		int cnt = 0;
		for (Intersectable *object : objects) {
			if(object->intersect(ray).t > 0 && cnt != 0)
				return true;
			cnt++;
		}
		return false;
	}



	vec3 trace(Ray ray, int depth = 0) {

		if(depth > 10){
			return La;
		}

		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La * 2 + Le * powf(dot(ray.dir, normalize(lightDirection - ray.start)), 10);

		vec3 outRadiance(0,0,0);

		if(hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light *light : lights) {
				for (int i = 0; i < controlPoints.size(); i++) {
					vec3 lightDir = vec3(controlPoints[i] - hit.position);
					float r = length(lightDir);

					Ray shadowRay(hit.position + hit.normal * epsilon, normalize(lightDir));
					float cosTheta = dot(hit.normal, lightDir);

					float cosBeta = dot(normalize(vec3(0,0,1)), lightDir);
					float omegaDelta = (M_PI * 0.62f * 0.62f / controlPoints.size()) * (cosBeta / powf(r, 2));

					if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
						vec3 halfway = normalize(-ray.dir + lightDir);
						float cosDelta = dot(hit.normal, halfway);
						outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta * omegaDelta;

						if (cosDelta > 0) {
							outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess) * omegaDelta;
						}
						outRadiance = outRadiance + trace(shadowRay, depth++) * omegaDelta;
					}

				}
			}

		}
		if(hit.material->type == REFLECTIVE){
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1,1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image) 
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN,0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
