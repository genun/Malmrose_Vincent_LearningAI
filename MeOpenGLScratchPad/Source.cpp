#include <qtgui\qapplication.h>
#include <Rendering\Renderer.h>
#include <IO\FileIO.h>
#include <Rendering\Helpers\Renderables.h>
#include <MyGame.h>
#include "Asteroids\AsterMain.h"

bool Test_Asteroids = true;

void main(int argumentCount, char* argumentVector[]){

	if (Test_Asteroids){
		AsterMain newGame;
		newGame.run();
	}
	
	else{
		QApplication app(argumentCount, argumentVector);
		MyGame game;

		if (!game.initialize())
			return;

		game.gameLoop();

		app.exec();
		game.shutdown();
	};
}
