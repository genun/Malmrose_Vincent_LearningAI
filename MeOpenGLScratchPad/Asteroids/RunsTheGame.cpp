#include "RunsTheGame.h"

RunTheGame::RunTheGame(){
	init();
}

void RunTheGame::init(){
	RunTheGame::drawBorder = false;
	meShip.position = Vector2d(200.0f, 200.0f);
	meShip.translation.mat[0][2] = 200.0f;
	meShip.translation.mat[1][2] = 200.0f;
	
	e.init();

	turret.init(meShip.translation);
	myLERPER.Lposition = Vector2d(50.0f, 50.0f);

	RunTheGame::SCREEN_WIDTH = 400;
	RunTheGame::SCREEN_HEIGHT = 400;

	Matrix3 rotate = Matrix3();
	rotate.Rotation(5);
	Matrix3 trans = Matrix3();
	trans.mat[0][2] = 250;
	trans.mat[1][2] = 250;
	Matrix3 nextOrbit = Matrix3();
	nextOrbit.mat[0][2] = 15;
	orbit.init(rotate, trans, nextOrbit, 0.2f, Vector2d(15, 15));
	profile.initialize();
	ScreenType = 0;
	width = SCREEN_WIDTH + 0.0f;
	height = SCREEN_HEIGHT + 0.0f;

	hp = 5;
	score = 0;
	for(unsigned int i = effect.effects.size(); i > 0; i--){
		effect.effects.erase(effect.effects.begin() + i - 1);
	}
	for(unsigned int i = e.enemies.size(); i > 0; i--){
		e.enemies.erase(e.enemies.begin() + i - 1);
	}
	meShip.rot = 0;
	meShip.rotation = Matrix3();
	meShip.velocity = Vector2d();
	win = false;
}

bool RunTheGame::MeUpdateFn(float dt){
	dt;
	MainUpdate(dt);

	if (ScreenType == 2){
		init();
		if (win){
			//TODO: notify AI of success
		}
		else{
			//TODO: notify AI of failure
		}
	}


	//if(ScreenType == 0){
	//	TitleUpdate(dt);
	//}
	//else if (ScreenType == 1){
	//	MainUpdate(dt);
	//}
	//else{
	//	EndUpdate(dt);
	//}
	return false;
}

void RunTheGame::MeDrawFn(Graphics& graphics){
	MainDraw(graphics);

	//if(ScreenType == 0){
	//	TitleDraw(graphics);
	//}
	//else if (ScreenType == 1){
	//	MainDraw(graphics);
	//}
	//else{
	//	EndDraw(graphics);
	//}
	//
	/*
	memoryInfo << BlockCount << local state.lCounts[_client_Block] << "\n"
	<< total bytes << localState.Lsizes[clientblock]
	<<Most Bytes Ever: << local state.lHighWaterCount;*/
	
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtMemState localState;
	_CrtMemCheckpoint(&localState);
	
	graphics.SetColor(RGB(100, 100, 100));
	graphics.DrawString(600, 650, "Current Block Size: ");
	DrawValue(graphics, 750, 650, localState.lCounts[_CLIENT_BLOCK]);
	graphics.DrawString(600, 700, "    Max Block Size: ");
	DrawValue(graphics, 750, 700, localState.lSizes[_CLIENT_BLOCK]);
	graphics.DrawString(600, 750, "Most Mem Allocated: ");
	DrawValue(graphics, 750, 750, localState.lHighWaterCount);
	graphics.SetColor(RGB(255, 255, 255));
}

void RunTheGame::MainUpdate(float dt){
	meShip.update(dt, drawBorder, width, height);
	time = time = timer.interval();
	profile.addEntry(time);

	e.update(dt, myBullet, effect, meShip, score, hp);
	
	if(Core::Input::IsPressed(Core::Input::KEY_RIGHT) || Core::Input::IsPressed('D')){	
		meShip.rot += 0.05f;
		meShip.rotation.Rotation(meShip.rot);
	}
	
	if(Core::Input::IsPressed(Core::Input::KEY_LEFT) || Core::Input::IsPressed('A')){
		meShip.rot -= 0.05f;
		meShip.rotation.Rotation(meShip.rot);
	}
	
	if(Core::Input::IsPressed(Core::Input::KEY_DOWN) || Core::Input::IsPressed('S')){
		meShip.velocity = meShip.velocity + (meShip.rotation * meShip.Speed);
	}
	
	if(Core::Input::IsPressed(Core::Input::KEY_UP) || Core::Input::IsPressed('W')){
		meShip.velocity = meShip.velocity + (meShip.rotation * (-1 * meShip.Speed));
	}

	//FLAMETHROWER WOOOO
	//if(Core::Input::IsPressed(Core::Input::BUTTON_RIGHT)){
	//	effect.add(ParticleEffects(rand.randomColor(200, 50, 50, 55), meShip));
	//}
	time = timer.interval();
	profile.addEntry(time);
	
	//myLERPER.update(dt);
	//time = timer.interval();
	//profile.addEntry(time);

	/*
	int mouseX, mouseY;
	Core::Input::GetMousePos(mouseX, mouseY);
	float x = mouseX - 0.0f;
	float y = mouseY - 0.0f;*/
	turret.update(dt, meShip.translation, meShip.rotation);
	time = timer.interval();
	profile.addEntry(time);
	
	if(myBullet.alive){
		myBullet.update(dt);
	}
	time = timer.interval();
	profile.addEntry(time);
	
	if(Core::Input::IsPressed(Core::Input::BUTTON_LEFT)){
		myBullet.init(Vector2d(turret.translation.mat[0][2], turret.translation.mat[1][2]), turret.rotation, meShip.velocity);
	}
	time = timer.interval();
	profile.addEntry(time);

	//orbit.update(dt);
	//time = timer.interval();
	//profile.addEntry(time);

	effect.update(dt);
	time = timer.interval();
	profile.addEntry(time);

	if(hp < 1){
		win = false;
		ScreenType = 2;
	}
	else if(score > 20){
		win = true;
		ScreenType = 2;
	}
}

void RunTheGame::MainDraw(Graphics& g){
	meShip.draw(g);
	time = timer.interval();
	profile.addEntry(time);

	e.draw(g);

	//orbit.draw(g, 1, 3, Vector2d(650, 650));
	//time = timer.interval();
	//profile.addEntry(time);
	//
	//myLERPER.draw(g);
	//time = timer.interval();
	//profile.addEntry(time);

	turret.draw(g);
	time = timer.interval();
	profile.addEntry(time);

	myBullet.draw(g);
	time = timer.interval();
	profile.addEntry(time);

	if(drawBorder){
		float width = SCREEN_WIDTH + 0.0f;
		float height = SCREEN_HEIGHT + 0.0f;
		g.DrawLine(width / 2, 0, width, height / 2);
		g.DrawLine(width, height /2, width / 2, height);
		g.DrawLine(width / 2, height, 0, height / 2);
		g.DrawLine(0, height / 2, width / 2, 0);
	}
	time = timer.interval();
	profile.addEntry(time);

	effect.draw(g);
	time = timer.interval();
	profile.addEntry(time);

	/*g.SetColor( RGB(0xff, 0xff, 0xff));
	g.DrawString(20, 10, "Use WASD to navigate the ship and Left click to fire a Missile, Right click to fire the flamethrower");
	g.DrawString(20, 25, "Use 1, 2, and 3 to alternate between warp, bounce, and collision");
	g.SetColor( RGB(255, 255, 255));
	DrawValue(g, 20, 45, meShip.currentTrans);
	time = timer.interval();
	profile.addEntry(time);
	*/
	g.SetColor( RGB(255, 255, 255));
	
	g.DrawString(15, 25, "Score: ");
	DrawValue(g, 55, 25, score);

	//g.DrawString(20, 25, "FPS");
	timer.stopTimer();
	//DrawValue(g, 55, 25, (1 / timer.getElapsedTime()));
	timer.startTimer();
	time = timer.interval();
	profile.addEntry(time);


}

#pragma region Title and end screen code
void RunTheGame::TitleUpdate(float dt){
	dt;
	if (Core::Input::IsPressed('P')){
		ScreenType = 1;
	}
}

void RunTheGame::TitleDraw(Graphics& g){
	g.SetColor(RGB(255, 150, 75));
	g.DrawString(20, 10, "Use WASD to navigate the ship and Left click to fire a Missile, Right click to fire the flamethrower");
	g.SetColor(RGB(255, 150, 75));
	g.DrawString(20, 25, "Use 1, 2, and 3 to alternate between warp, bounce, and collision");
	g.SetColor(RGB(255, 150, 75));
	g.DrawString(20, 40, "Press p to start the game");
	g.DrawString(20, 65, "You win by killing 21 enemies");
	g.SetColor(RGB(255, 255, 255));
}

void RunTheGame::EndUpdate(float dt){
	dt;
	if(Core::Input::IsPressed('R')){
		init();
	}
}

void RunTheGame::EndDraw(Graphics& g){
	g;
	if(win){
		g.SetColor(RGB(255, 255, 0));
		g.DrawString(400, 400, "YOU WIN!");
	}
	else{
		g.SetColor(RGB(0, 255, 255));
		g.DrawString(400, 400, "YOU LOSE!");
	}
	g.SetColor(RGB(255, 0, 0));
	g.DrawString(400, 430, "Your Score is: ");
	DrawValue(g, 550, 430, score);

	g.DrawString(400, 460, "Press R to Restart");


}
#pragma endregion