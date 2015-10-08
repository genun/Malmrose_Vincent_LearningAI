#include "RunsTheGame.h"
#include "Engine.h"
#include "Core.h"
#include "Random.h"
#include "Particle.h"
#include "DrawV.h"
#include "ParticleEffects.h"
#include <time.h>
#include "Assertion.h"
#include "Logger.h"
#include "MemTrack.h"
#include "AsterMain.h"
#include <glm/glm/glm.hpp>

RunTheGame game;
Random randy;


bool MeUpdateFn(float dt){
	dt;
	if(Core::Input::IsPressed(Core::Input::KEY_ESCAPE)){
		return true;
	}
	game.MeUpdateFn(dt);
	return false;
}

void MeDrawFn(Graphics& graphics){
	game.MeDrawFn(graphics);
}

void runWithCore(){
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	_CrtMemState localState;
	_CrtMemCheckpoint(&localState);

	int x;
	x = 1;

	//Had to use a variable because the optimizer was messing it up otherwise.
	//ASSERT(x == 4);

	srand((unsigned)time(NULL));
	game = RunTheGame();
	Core::Init("PEW PEW spaceships", game.SCREEN_WIDTH, game.SCREEN_HEIGHT);
	Core::RegisterUpdateFn(MeUpdateFn);
	Core::RegisterDrawFn(MeDrawFn);
	Core::GameLoop();
	Core::Shutdown();
}

void AsterMain::run()
{
	runWithCore();
}
