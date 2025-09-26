// src/app.module.ts
import { Module } from '@nestjs/common';
import { MulterModule } from '@nestjs/platform-express';
import { TaggingController } from '../tagging.controller';
import { TaggingService } from '../tagging.service';

@Module({
  imports: [
    MulterModule.register({
      dest: './uploads',
    }),
  ],
  controllers: [TaggingController],
  providers: [TaggingService],
})
export class AppModule {}
