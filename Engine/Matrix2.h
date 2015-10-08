#ifndef MATRIX2_H
#define MATRIX2_H

#include "Vector2d.h"

struct Matrix2{
	float mat[2][2];
	Matrix2(){
		mat[0][0] = 1.0f;
		mat[1][0] = 0.0f;
		mat[0][1] = 0.0f;
		mat[1][1] = 1.0f;
	}
	Matrix2(Vector2d v1, Vector2d v2){
		mat[0][0] = v1.x;
		mat[1][0] = v2.x;
		mat[0][1] = v1.y;
		mat[1][1] = v2.y;
	}
	inline void init(float n1, float n2, float n3, float n4){
		mat[0][0] = n1;
		mat[1][0] = n2;
		mat[0][1] = n3;
		mat[1][1] = n4;
	}

	inline void Rotation(float angle){
		float c = cos(angle);
		float s = sin(angle);

		mat[0][0] = c;
		mat[1][0] = -s;
		mat[0][1] = s;
		mat[1][1] = c;
	}
	
	operator float*(){
		return &mat[0][0];
	}

	inline void Scale(float scale){
		mat[0][0] = scale;
		mat[1][1] = scale;
	}

	inline void ScaleX(float scale){
		mat[0][0] = scale;
	}

	inline void ScaleY(float scale){
		mat[1][1] = scale;
	}
};



inline Matrix2 operator*(const Matrix2& m1, const Matrix2& m2){
	Matrix2 myMatrix;
	float n1 = m1.mat[0][0] * m2.mat[0][0] + m1.mat[1][0] * m2.mat[0][1];
	float n2 = m1.mat[0][1] * m2.mat[0][0] + m1.mat[1][1] * m2.mat[0][1];
	float n3 = m1.mat[0][0] * m2.mat[1][0] + m1.mat[1][0] * m2.mat[1][1];
	float n4 = m1.mat[0][1] * m2.mat[1][0] + m1.mat[1][1] * m2.mat[1][1];
	myMatrix.init(n1, n2, n3, n4);
	return myMatrix;
}

inline Vector2d operator*(const Matrix2& m1, const Vector2d& v2){
	Vector2d newVect;
	newVect.x = m1.mat[0][0] * v2.x + m1.mat[1][0] * v2.y;
	newVect.y = m1.mat[0][1] * v2.x + m1.mat[1][1] * v2.y;
	return newVect;
}

inline Vector2d operator*(const Vector2d& v2, const Matrix2& m1){
	Vector2d newVect;
	newVect.x = m1.mat[0][0] * v2.x + m1.mat[1][0] * v2.y;
	newVect.y = m1.mat[0][1] * v2.x + m1.mat[1][1] * v2.y;
	return newVect;
}

#endif