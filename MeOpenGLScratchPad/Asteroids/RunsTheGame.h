#ifndef RUNTHEGAME_H
#define RUNTHEGAME_H
#include "Engine.h"
#include "Core.h"
#include "SpaceShip.h"
#include "Monster.h"
#include "Turret.h"
#include "Bullet.h"
#include "orbiter.h"
#include "DrawV.h"
#include <sstream>
#include "ParticleManager.h"
#include "Random.h"
#include "hr_timer.h"
#include "profiler.h"
#include "enemyManager.h"
#include "MemTrack.h"
using Core::Graphics;

struct RunTheGame{
	RunTheGame();
	bool win;
	bool drawBorder;
	int SCREEN_WIDTH ;
	int SCREEN_HEIGHT;
	int ScreenType;
	int score, hp;
	float width;
	float height;
	double time;

	SpaceShip meShip;
	Monster myLERPER;
	Turret turret;
	bullet myBullet;
	orbiter orbit;
	ParticleManager effect;
	Random rand;
	CStopWatch timer;
	profiler profile;
	enemyManager e;

	bool MeUpdateFn(float dt);
	void MeDrawFn  (Graphics& graphics);
	void Init();
	
	void TitleUpdate(float dt);
	void TitleDraw(Graphics& graphics);
	
	void MainUpdate(float dt);
	void MainDraw(Graphics& graphics);
	
	void EndUpdate(float dt);
	void EndDraw(Graphics& graphics);

	void init();
};
	

#endif