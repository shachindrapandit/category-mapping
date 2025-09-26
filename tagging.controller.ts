import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
  Body,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { TaggingService } from './tagging.service';
import { Express } from 'express';  // Correct import

@Controller('tagging')
export class TaggingController {
  constructor(private readonly taggingService: TaggingService) {}

  @Post('predict')
  @UseInterceptors(FileInterceptor('image'))
  async predict(
    @UploadedFile() file: Express.Multer.File,  // Correct type here
    @Body('title') title: string,
    @Body('description') description: string,
  ) {
    const filePath = file.path; // available if using diskStorage
    return this.taggingService.predictCategory(filePath, title, description);
  }
}
