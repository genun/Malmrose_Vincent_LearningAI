#include "Turret.h"
#include "Vector2d.h"

void Turret::init(Matrix3 mat){
	Points[0] = Vector2d(-6, -6);
	Points[1] = Vector2d(-6, 6);
	Points[2] = Vector2d(6, 6);
	Points[3] = Vector2d(6, -6);
	Points[4] = Vector2d(3, -6);
	Points[5] = Vector2d(3, - 12);
	Points[6] = Vector2d(-3, -12);
	Points[7] = Vector2d(-3, -6);
	Points[8] = Vector2d(-6, -6);

	transformation = mat;
};


void Turret::draw(Core::Graphics& g){
	const unsigned int NUM_LINES = sizeof(Points) / sizeof(*Points);
	for(unsigned int i = 0; i < NUM_LINES - 1; i++){
		const Vector2d& p1 = transformation * Points[i];
		const Vector2d& p2 = transformation * Points[(i+1) % NUM_LINES];
		g.DrawLine(p1.x, p1.y, p2.x, p2.y);
	}
}

void Turret::update(float dt, Matrix3 ShipTrans, Matrix3 shipRotation){
	dt;
	translation = ShipTrans;
	//MousePoint - 

	Vector2d dir = Vector2d(ShipTrans.mat[0][2], ShipTrans.mat[1][2]);
	dir = dir.Normalized();
	Vector2d otherDir = dir.PerpCCW();
	rotation.mat[0][0] = -otherDir.x;
	rotation.mat[1][0] = -otherDir.y;
	rotation.mat[0][1] = -dir.x;
	rotation.mat[1][1] = -dir.y;
	rotation = shipRotation;
	transformation = translation * shipRotation;
};
