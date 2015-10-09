#include "SpaceShip.h"
#include "Core.h"

#pragma region drawing and initialization
SpaceShip::SpaceShip(){
	SpaceShip::typeChange = 1;
	rot = 0;
	Speed = Vector2d(0, 0.10f);
}

Vector2d ShipPoints[17] = {
	Vector2d(-00.0f, -20.0f),
	Vector2d(+05.0f, -10.0f),
	Vector2d(+10.0f, -05.0f),
	Vector2d(+15.0f, +05.0f),
	Vector2d(+15.0f, +15.0f),
	Vector2d(+13.5f, +11.5f),
	Vector2d(+09.0f, +07.5f),
	Vector2d(+05.0f, +05.0f),
	Vector2d(+00.0f, +15.0f),
	Vector2d(-05.0f, +05.0f),
	Vector2d(-09.0f, +07.5f),
	Vector2d(-13.5f, +11.5f),
	Vector2d(-15.0f, +15.0f),
	Vector2d(-15.0f, +05.0f),
	Vector2d(-10.0f, -05.0f),
	Vector2d(-05.0f, -10.0f),
	Vector2d(-00.0f, -20.0f),
};

void SpaceShip::draw(Core::Graphics& g){
	const unsigned int NUM_LINES = sizeof(ShipPoints) / sizeof(*ShipPoints);
	for(unsigned int i = 0; i < NUM_LINES - 1; i++){
		currentTrans = translation * rotation;
		const Vector2d& p1 = currentTrans * ShipPoints[i];
		const Vector2d& p2 = currentTrans * ShipPoints[(i+1) % NUM_LINES];
		g.DrawLine(p1.x, p1.y, p2.x, p2.y);
	}
}

bool inside = false;
#pragma endregion

void SpaceShip::update(float dt, bool& drawBorder, float width, float height){
	dt;
	drawBorder = false;

	velocity = velocity * 0.98f;
	translation.mat[0][2] = translation.mat[0][2] + velocity.x;
	translation.mat[1][2] = translation.mat[1][2] + velocity.y;
	
	position = Vector2d(translation.mat[0][2], translation.mat[1][2]);

	position = position + SpaceShip::velocity;

#pragma region barrier changes

	//Change barrier type
	//if(Core::Input::IsPressed('1')){
	//	SpaceShip::typeChange = 1;
	//}
	//
	//else if(Core::Input::IsPressed('2')){
	//	SpaceShip::typeChange = 2;
	//}
	//
	//else if(Core::Input::IsPressed('3')){
	//	inside = false;
	//	SpaceShip::typeChange = 3;
	//}


	//
	//if(typeChange == 2){
	//	wallBounce();
	//}

	//else if(typeChange == 3){
	//	drawBorder = true;
	//	/*
	//		middle top,
	//		middle right,
	//		middle bottom,
	//		middle left
	//	*/
	//	Vector2d v1 = Vector2d(width / 2, 0); 
	//	Vector2d v2 = Vector2d(width, height / 2);
	//	Vector2d v3 = Vector2d(width / 2, height);
	//	Vector2d v4 = Vector2d(0, height /2);

	//	Vector2d w1 = v2 - v1;
	//	Vector2d w2 = w1.PerpCW();
	//	Vector2d w3 = v4 - v3;
	//	Vector2d w4 = w3.PerpCW();

	//	Vector2d s1 = position - v1;
	//	Vector2d s2 = position - v3;
	//	if (!inside){
	//		wallBounce();
	//	}
	//	if(s1.Dot(w1.PerpCCW()) >= 0  && s1.Dot(w2.PerpCCW()) >= 0 && s2.Dot(w3.PerpCCW()) >= 0 && s2.Dot(w4.PerpCCW()) >= 0){
	//		inside = true;
	//	}

	//	if(s1.Dot(w1.PerpCCW()) < 0 && inside){
	//		bounce(w1);
	//	}
	//	if(s1.Dot(w2.PerpCCW()) < 0 && inside){
	//		bounce(w2);
	//	}
	//	if(s2.Dot(w3.PerpCCW()) < 0 && inside){
	//		bounce(w3);
	//	}
	//	if(s2.Dot(w4.PerpCCW()) < 0 && inside){
	//		bounce(w4);
	//	}
	//}
#pragma endregion

	//Let ship warp accross the screen
	if (position.x < -20){
		position.x = 400;
		translation.mat[0][2] = 400;
	}
	if (position.x > 420){
		position.x = 0;
		translation.mat[0][2] = 0;
	}
	if (position.y < -20){
		position.y = 400;
		translation.mat[1][2] = 400;
	}
	if (position.y  > 420){
		position.y = 0;
		translation.mat[1][2] = 0;
	}
}

#pragma region Bouncy
void SpaceShip::wallBounce(){
		if (position.x < 20){
			velocity.x = velocity.x * -1;
		}
		if (position.x > 780){	
			velocity.x = velocity.x * -1;
		}
		if (position.y < 20){
			velocity.y = velocity.y * -1;
		}
		if (position.y  > 780){
			velocity.y = velocity.y * -1;
		}
}

void SpaceShip::bounce(Vector2d wall){
	Vector2d v1 = wall.PerpCCW();
	Vector2d v2 = v1.Normalized();
	float v4 = velocity.Dot(v2);
	//Vector2d v2 = wall.Dot(velocity);
	Vector2d v5 = v4 * -1 * v2;
	velocity = velocity + v5 + v5;
}
#pragma endregion