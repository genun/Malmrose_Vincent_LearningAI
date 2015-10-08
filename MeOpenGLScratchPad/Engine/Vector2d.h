#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <math.h>

struct Vector2d{
	float x, y, z;
	Vector2d(float x = 0.0f, float y = 0.0f, float z = 0.0f): x(x), y(y){ z = 0.0f;}

	friend Vector2d operator+(const Vector2d& v1, const Vector2d& v2);
	friend Vector2d operator*(const float& f1, const Vector2d& v2);
	friend Vector2d operator*(const Vector2d& v1, const float& f2);
	friend Vector2d operator/(const Vector2d& v1, const float& f2);
	friend Vector2d operator-(const Vector2d& v1, const Vector2d& v2);/*
	friend Matrix2 operator*(const Matrix2& m1, const Matrix2& m2);
	friend Matrix2 operator*(const Matrix2& m1, const Vector2d& v2);
	friend Matrix2 operator*(const Vector2d& v1, const Matrix2& m2);*/
	
	inline Vector2d FakeLERP(float scalar){
		return Vector2d(x * scalar, y * scalar);
	}
	
	inline Vector2d LERP(float scalar, Vector2d v2){
		Vector2d v3((1 - scalar) * x, (1 - scalar) * y);
		Vector2d v4(scalar * v2.x, scalar * v2.y);
		return v3 + v4;
	}

	operator float*(){
		return &x;
	}

	inline float Length(){
		return sqrt(x*x + y*y);
	}

	inline float LengthSquared(){
		return Length() * Length();
	}

	inline Vector2d Normalized(){
		if(Length() == 0){
			return 0.000000001f;
		}
		else{
			return Vector2d(x / Length(), y / Length());
		} 
	}

	inline Vector2d PerpCW(){

		Vector2d v1 =Vector2d(y, -1 * x);
		return v1;
	}

	inline Vector2d PerpCCW(){
		return Vector2d(y * -1, x);
	}



	inline float Dot(Vector2d other){
		return x * other.x + y * other.y;
	}



	inline Vector2d Cross(Vector2d other){
		return x * other.y - y * other.x;
	}
};


inline Vector2d Dot(Vector2d v1, Vector2d v2){
	return v1.x * v2.x + v1.y + v2.y;
}

inline Vector2d operator-(const Vector2d& v1, const Vector2d& v2){
		return Vector2d(v1.x - v2.x, v1.y - v2.y);
}

inline Vector2d operator*(const float& f1, const Vector2d& v2){
	return Vector2d(v2.x * f1, v2.y * f1);
	
}

inline Vector2d operator*(const Vector2d& v1, const float& f2){
	return Vector2d(v1.x * f2, v1.y * f2);
}

inline Vector2d operator/(const Vector2d& v1, const float& f2){
	return Vector2d(v1.x / f2, v1.y / f2);
}

inline std::ostream& operator<<(std::ostream& stream, const Vector2d& right){
	std::cout << "{" << right.x << ", " << right.y << "}";
	return stream;
}

inline Vector2d operator+(const Vector2d& v1, const Vector2d& v2){
	return Vector2d(v1.x + v2.x, v1.y + v2.y);
}


#endif