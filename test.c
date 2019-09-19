#include <stdio.h>

int main()
{
	int temp = 0;

	scanf("%d", &temp);
	printf("temp=%d\n", temp);

	switch (temp)
	{	
		case 4:
			printf("case 4\n");

		case 2:
			printf("case 2\n");
			
		case 1:
			printf("case 1\n");
			break;
		default:
			printf("default\n");
			break;
	}
	return temp;
}
