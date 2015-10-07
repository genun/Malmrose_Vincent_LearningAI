#include <qtgui\qapplication.h>
#include <Rendering\Renderer.h>
#include <IO\FileIO.h>
#include <Rendering\Helpers\Renderables.h>
#include <MyGame.h>

void main(int argumentCount, char* argumentVector[]){
	QApplication app(argumentCount, argumentVector);
	MyGame game;
	
	if( ! game.initialize())
		return;

	game.gameLoop();

	app.exec();
	game.shutdown();
}
