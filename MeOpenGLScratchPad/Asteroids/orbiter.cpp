#include "orbiter.h"


void orbiter::draw(Core::Graphics& g, float numOrbits, float totalOrbits, Vector2d position){
	//FinalTrans;
	//ORBIT TRANSLATE
	Matrix3 orbitTrans = Matrix3();
	float offset =  1 * (totalOrbits - numOrbits - 2);
	Vector2d vect = Vector2d(-20, -20)  * offset;
	orbitTrans.Translation(vect);
	total = orbitTrans;

	//ROTATE
	rotate.Rotation(rotateNumber * (1 - offset));
	total = total * rotate;
	

	////TRANSLATE TO ORIGIN
	Matrix3 FinalTrans;
	FinalTrans.Translation(position);
	//total = total * FinalTrans;
	//total.mat[0][2] = total.mat[0][2] + FinalTrans.mat[0][2];
	//total.mat[1][2] = total.mat[1][2] + FinalTrans.mat[1][2];
	total.mat[0][2] = total.mat[0][2] + position.x - orbitTrans.mat[0][2];
	total.mat[1][2] = total.mat[1][2] + position.y - orbitTrans.mat[1][2];

	//SCALE POINTS
	Matrix3 scaleMatrix;
	scaleMatrix.Scale(2/numOrbits);
	Vector2d tempPoints[6];
	tempPoints[0] = scaleMatrix * points[0];
	tempPoints[1] = scaleMatrix * points[1];
	tempPoints[2] = scaleMatrix * points[2];
	tempPoints[3] = scaleMatrix * points[3];
	tempPoints[4] = scaleMatrix * points[4];
	tempPoints[5] = scaleMatrix * points[5];
	
	// + (pos * offset)
	const unsigned int NUM_LINES = sizeof(tempPoints) / sizeof(*tempPoints);
	for(unsigned int i = 0; i < NUM_LINES - 1; i++){
		const Vector2d& p1 = total * (tempPoints[i]                 + (pos * offset));
		const Vector2d& p2 = total * (tempPoints[(i+1) % NUM_LINES] + (pos * offset));
		g.DrawLine(p1.x, p1.y, p2.x, p2.y);
	}

	//Works but is rotating off the top left corner of the planet house instead of the actual center. Still works though.
	if(numOrbits < totalOrbits){
		Vector2d v1 = total * (tempPoints[0] + (pos * offset));
		draw(g, numOrbits + 1, totalOrbits, v1);//total.mat[0][2], total.mat[1][2]));
	}
}


void orbiter::update(float dt){
	dt;
	rotateNumber += dt;
	if(rotateNumber > 3000){
		rotateNumber = 0.2f;
	}
	rotate.Rotation(rotateNumber);
}

void orbiter::init(Matrix3 rot, Matrix3 t, Matrix3 nextOr, float rNum,  Vector2d p){
	rotate = rot;
	trans = t;
	total = trans * rotate;
	nextOrbit = nextOr;
	pos = p;
	points[0] = Vector2d(6, 6);
	points[1] = Vector2d(-6 , 6);
	points[2] = Vector2d(-6 , -6);
	points[3] = Vector2d(0 , -9);
	points[4] = Vector2d(6, -6);
	points[5] = Vector2d(6, 6);
	rotateNumber = rNum;
}
